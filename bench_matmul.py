"""INT8 / INT4 GEMM under speculative-decoding verification shapes.

Part A: fine-grained M sweep (the "staircase") per BLOCK_M -> tile utilisation.
Part B: (batch B, draft length k) grid on Llama-3-8B linear shapes, M = B*(k+1),
        oracle config per shape vs fixed configs, plus MaxPad (pad to k=8).
"""
import itertools, json, math
import torch, triton
from triton.testing import do_bench
from kernels import *

dev = "cuda"
torch.manual_seed(0)
bench = lambda f: do_bench(f, warmup=10, rep=40, return_mode="median")  # ms

LLAMA3_8B = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}


def make_weights(N, K):
    w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
    w_q, sw = quantize_weight_int8(w)
    wp, sc, _ = quantize_weight_int4(w, 128)
    return w, w_q, sw, wp, sc


def runner(kind, M, N, K, W, cfg, x_cache={}):
    w, w_q, sw, wp, sc = W
    key = (M, K)
    if key not in x_cache:
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        a_q, sa = quantize_act_int8(x)
        x_cache[key] = (x, a_q, sa, torch.empty(M, N, device=dev, dtype=torch.float16))
    x, a_q, sa, _ = x_cache[key]
    out = torch.empty(M, N, device=dev, dtype=torch.float16)
    if kind == "fp16_cublas":
        return lambda: torch.matmul(x, w.T, out=out)
    BM, BN, BK, nw, ns = cfg
    if kind == "w8a8":
        return lambda: w8a8_matmul(a_q, sa, w_q, sw, out=out, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                   num_warps=nw, num_stages=ns)
    if kind == "w4a16":
        return lambda: w4a16_matmul(x, wp, sc, out=out, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=min(BK, 128),
                                    num_warps=nw, num_stages=ns)
    if kind == "fp16_cublas":
        return lambda: torch.matmul(x, w.T, out=out)
    raise ValueError(kind)


def try_bench(kind, M, N, K, W, cfg):
    try:
        f = runner(kind, M, N, K, W, cfg)
        f()
        return bench(f)
    except Exception as e:      # out of shared memory etc.
        return None


out = {"device": torch.cuda.get_device_name(), "partA": [], "partB": []}

# ---------------- Part A: staircase on o_proj (4096x4096) ----------------------------
N, K = LLAMA3_8B["o"]
W = make_weights(N, K)
Ms = list(range(1, 49)) + list(range(56, 161, 8)) + [192, 224, 256, 288]
for M in Ms:
    row = {"M": M}
    row["fp16_cublas"] = try_bench("fp16_cublas", M, N, K, W, None)
    for BM in (16, 32, 64, 128):
        row[f"w8a8_BM{BM}"] = try_bench("w8a8", M, N, K, W, (BM, 64, 128, 4, 3))
        row[f"w4a16_BM{BM}"] = try_bench("w4a16", M, N, K, W, (BM, 64, 64, 4, 3))
        row[f"util_BM{BM}"] = M / (math.ceil(M / BM) * BM)
    out["partA"].append(row)
    print("A", M, {k: (round(v * 1000, 1) if isinstance(v, float) and "util" not in k else v)
                   for k, v in row.items() if k != "M"}, flush=True)

# ---------------- Part B: (B, k) grid, all Llama-3-8B linears ----------------------
configs = [(bm, bn, bk, 4, 3) for bm in (16, 32, 64, 128) for bn in (64, 128) for bk in (64, 128)]
configs += [(128, 128, 64, 8, 3), (64, 128, 128, 8, 3)]
del W
torch.cuda.empty_cache()
for name, (N, K) in LLAMA3_8B.items():
    W = make_weights(N, K)
    cache = {}
    for kind in ("w8a8", "w4a16", "fp16_cublas"):
        for B in (1, 2, 4, 8, 16, 32):
            for k in (0, 2, 4, 8):           # k=0 -> plain decode baseline
                M = B * (k + 1)
                if kind == "fp16_cublas":
                    t_best, cfg_best, t16, t64 = try_bench(kind, M, N, K, W, None), None, None, None
                else:
                    ts = {}
                    for c in configs:
                        if (kind, c, M) not in cache:
                            cache[(kind, c, M)] = try_bench(kind, M, N, K, W, c)
                        ts[c] = cache[(kind, c, M)]
                    valid = {c: t for c, t in ts.items() if t is not None}
                    cfg_best = min(valid, key=valid.get)
                    t_best = valid[cfg_best]
                    t16 = ts.get((16, 64, 128, 4, 3))
                    t64 = ts.get((64, 128, 64, 4, 3))
                rec = dict(layer=name, N=N, K=K, kind=kind, B=B, k=k, M=M, t_best_ms=t_best,
                           best_cfg=cfg_best, t_BM16_ms=t16, t_BM64_ms=t64)
                out["partB"].append(rec)
                print("B", name, kind, B, k, M, round(t_best * 1000, 1), cfg_best, flush=True)
    del W
    torch.cuda.empty_cache()

json.dump(out, open("results/matmul.json", "w"), indent=1, default=str)
