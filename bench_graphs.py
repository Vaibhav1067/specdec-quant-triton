"""CUDA Graph strategies for dynamic verification shapes with quantized kernels.

A stack of L Llama-3-8B-shaped layers (W8A8 QKV/O/gate_up/down + INT8 paged
attention) is driven by a random trace of (batch B, draft length k).

  eager     : exact shapes, plain Triton launches (no graph)
  allshapes : one graph per (B, k) captured up front           -> 0 % waste
  maxpad    : one graph per B, always verifies k_max=8 tokens  -> padding waste
  plan      : ONE graph at (B_max, k_max); real (M, B, q_len) written to a
              GPU plan buffer before replay, kernels early-exit (PolyGraph idea)
"""
import json, time, random
import numpy as np
import torch, triton
from kernels import *

dev = "cuda"
L = 4
HID, QKV, FF = 4096, 6144, 14336
H, H_KV, D = 32, 8, 128
B_MAX, KS = 16, (2, 4, 8)
QL_MAX = max(KS) + 1
M_MAX = B_MAX * QL_MAX
CTX = 1024
torch.manual_seed(0)

layers = []
for _ in range(L):
    lw = {}
    for name, (N, K) in {"qkv": (QKV, HID), "o": (HID, HID), "gu": (2 * FF, HID), "down": (HID, FF)}.items():
        lw[name] = quantize_weight_int8(torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02)
    lw["kv"] = make_paged_kv(B_MAX, [CTX] * B_MAX, H_KV, D, seed=len(layers))
    layers.append(lw)

# static activation buffers (max size) -- graphs need fixed addresses
x_in = quantize_act_int8(torch.randn(M_MAX, HID, device=dev, dtype=torch.float16))
x_ff = quantize_act_int8(torch.randn(M_MAX, FF, device=dev, dtype=torch.float16))
buf = {n: torch.empty(M_MAX, N, device=dev, dtype=torch.float16)
       for n, N in {"qkv": QKV, "o": HID, "gu": 2 * FF, "down": HID}.items()}
q_buf = torch.randn(B_MAX, QL_MAX, H, D, device=dev, dtype=torch.float16)
o_buf = torch.zeros_like(q_buf)
plan = torch.zeros(8, dtype=torch.int32, device=dev)          # [M, B, q_len, ...]
plan_host = torch.zeros(8, dtype=torch.int32).pin_memory()
import os
BM = int(os.environ.get("BLOCK_M", 16))
CFG = dict(BLOCK_M=BM, BLOCK_N=64, BLOCK_K=128)


def step(B, ql, use_plan=False):
    """One verification forward pass for B sequences x ql tokens."""
    M = B * ql
    kw = dict(plan=plan, grid_m_max=M_MAX) if use_plan else {}
    for lw in layers:
        a, sa = (x_in[0], x_in[1]) if use_plan else (x_in[0][:M], x_in[1][:M])
        w8a8_matmul(a, sa, *lw["qkv"], out=buf["qkv"], **CFG, **kw)
        kc, vc, ks, vs, bt, cl = lw["kv"]
        if use_plan:
            paged_attention(q_buf, kc, vc, ks, vs, bt, cl, QL_MAX, QPP=QL_MAX, plan=plan[1:], out=o_buf)
        else:
            paged_attention(q_buf[:B], kc, vc, ks, vs, bt[:B], cl[:B], ql, QPP=ql, out=o_buf[:B])
        w8a8_matmul(a, sa, *lw["o"], out=buf["o"], **CFG, **kw)
        w8a8_matmul(a, sa, *lw["gu"], out=buf["gu"], **CFG, **kw)
        a2, sa2 = (x_ff[0], x_ff[1]) if use_plan else (x_ff[0][:M], x_ff[1][:M])
        w8a8_matmul(a2, sa2, *lw["down"], out=buf["down"], **CFG, **kw)


def set_plan(B, ql):
    plan_host[0], plan_host[1], plan_host[2] = B * ql, B, ql
    plan.copy_(plan_host, non_blocking=True)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=POOL):
        fn()
    return g


# warm up / compile every specialisation used by eager
for B in range(1, B_MAX + 1):
    for k in KS:
        step(B, k + 1)
torch.cuda.synchronize()
POOL = torch.cuda.graph_pool_handle()

res = {"config": dict(L=L, B_MAX=B_MAX, KS=KS, CTX=CTX, M_MAX=M_MAX)}
t0 = time.perf_counter()
g_all = {(B, k): capture(lambda B=B, k=k: step(B, k + 1)) for B in range(1, B_MAX + 1) for k in KS}
torch.cuda.synchronize()
res["allshapes_build_s"], res["allshapes_graphs"] = time.perf_counter() - t0, len(g_all)

t0 = time.perf_counter()
g_pad = {B: capture(lambda B=B: step(B, QL_MAX)) for B in range(1, B_MAX + 1)}
torch.cuda.synchronize()
res["maxpad_build_s"], res["maxpad_graphs"] = time.perf_counter() - t0, len(g_pad)

set_plan(B_MAX, QL_MAX)
t0 = time.perf_counter()
g_plan = capture(lambda: step(B_MAX, QL_MAX, use_plan=True))
torch.cuda.synchronize()
res["plan_build_s"], res["plan_graphs"] = time.perf_counter() - t0, 1

strategies = {
    "eager": lambda B, k: step(B, k + 1),
    "allshapes": lambda B, k: g_all[(B, k)].replay(),
    "maxpad": lambda B, k: g_pad[B].replay(),
    "plan": lambda B, k: (set_plan(B, k + 1), g_plan.replay()),
}


def run_trace(seed, n=300, fixed_k=None):
    rng = random.Random(seed)
    trace = [(rng.randint(1, B_MAX), fixed_k if fixed_k else rng.choice(KS)) for _ in range(n)]
    out = {}
    for name, fn in strategies.items():
        for B, k in trace[:20]:
            fn(B, k)
        torch.cuda.synchronize()
        lat = []
        for B, k in trace:
            t = time.perf_counter()
            fn(B, k)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1e3)
        lat = np.array(lat)
        useful = sum(B * (k + 1) for B, k in trace)
        computed = sum(B * QL_MAX for B, k in trace) if name == "maxpad" else useful
        out[name] = dict(p50_ms=float(np.median(lat)), p99_ms=float(np.percentile(lat, 99)),
                         mean_ms=float(lat.mean()), waste_pct=100 * (computed - useful) / useful,
                         tok_per_s=useful / (lat.sum() / 1e3))
        print(seed, fixed_k, name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    return out


res["mixed"] = [run_trace(s) for s in (0, 1, 2)]
res["fixed_k"] = {k: run_trace(10 + k, fixed_k=k) for k in KS}

# correctness: plan-graph output == exact eager output for every (B, k)
max_err = 0.0
for B in range(1, B_MAX + 1):
    for k in KS:
        step(B, k + 1)
        ref = {n: buf[n][:B * (k + 1)].clone() for n in buf}
        ref_o = o_buf[:B, :k + 1].clone()
        for t in list(buf.values()) + [o_buf]:
            t.zero_()
        set_plan(B, k + 1)
        g_plan.replay()
        torch.cuda.synchronize()
        for n in buf:
            max_err = max(max_err, (buf[n][:B * (k + 1)].float() - ref[n].float()).abs().max().item())
        max_err = max(max_err, (o_buf[:B, :k + 1].float() - ref_o.float()).abs().max().item())
res["plan_vs_eager_max_abs_diff"] = max_err
print("plan vs eager max diff", max_err)
json.dump(res, open("results/graphs.json" if BM == 16 else f"results/graphs_bm{BM}.json", "w"), indent=1)
