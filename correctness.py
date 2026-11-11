"""Correctness of the INT8/INT4 kernels under dynamic (speculative) shapes."""
import json, sys
import torch, triton
from kernels import *
from kernels import _w8a8_kernel, _w4a16_kernel

torch.manual_seed(0)
dev = "cuda"
results = {}
fails = []


def check(name, ok, info=""):
    if not ok:
        fails.append(f"{name} {info}")


def n_compiled(jitfn):
    """Number of specialised binaries Triton has compiled for this kernel."""
    try:
        caches = jitfn.device_caches
        return sum(len(v[0]) for v in caches.values())
    except Exception:
        return sum(len(v) for v in jitfn.cache.values())


# ---------------- W8A8: every M from 1..300, every BLOCK_M -------------------
N, K = 1000, 520            # deliberately not multiples of the tile sizes
w = torch.randn(N, K, device=dev, dtype=torch.float16)
w_q, sw = quantize_weight_int8(w)
x_all = torch.randn(300, K, device=dev, dtype=torch.float16)
max_err = 0.0
for BM in (16, 32, 64, 128):
    for M in range(1, 301):
        a_q, sa = quantize_act_int8(x_all[:M])
        out, _ = w8a8_matmul(a_q, sa, w_q, sw, BLOCK_M=BM)
        ref = (a_q.float() @ w_q.float().T) * sa[:, None] * sw[None, :]   # exact int math
        err = ((out.float() - ref).abs() / (ref.abs() + 1)).max().item()
        max_err = max(max_err, err)
        check("w8a8", err < 2e-3, f"BM={BM} M={M} err={err}")
results["w8a8_exhaustive_M1-300_x4tiles_max_rel_err"] = max_err
results["w8a8_binaries_compiled_for_M1-300_x4tiles"] = n_compiled(_w8a8_kernel)

# quantisation error vs fp16 reference (model-level accuracy, not a kernel bug)
a_q, sa = quantize_act_int8(x_all)
out, _ = w8a8_matmul(a_q, sa, w_q, sw)
ref16 = x_all.float() @ w.float().T
results["w8a8_vs_fp16_rel_fro_err"] = ((out.float() - ref16).norm() / ref16.norm()).item()

# ---------------- W8A8 plan-guarded fixed grid (CUDA-graph mode) --------------
MMAX = 288
a_q, sa = quantize_act_int8(x_all[:MMAX])
plan = torch.zeros(8, dtype=torch.int32, device=dev)
stale_ok = True
for M in (1, 3, 5, 9, 15, 16, 17, 27, 45, 81, 144, 288):
    plan[0] = M
    out = torch.full((MMAX, N), 12345.0, device=dev, dtype=torch.float16)  # sentinel
    w8a8_matmul(a_q, sa, w_q, sw, out=out, plan=plan, grid_m_max=MMAX, BLOCK_M=16)
    ref = (a_q[:M].float() @ w_q.float().T) * sa[:M, None] * sw[None, :]
    err = ((out[:M].float() - ref).abs() / (ref.abs() + 1)).max().item()
    check("w8a8_plan", err < 2e-3, f"M={M} err={err}")
    # rows beyond the active M must not be touched
    stale_ok &= bool((out[M:] == 12345.0).all().item())
results["w8a8_plan_rows_beyond_M_untouched"] = stale_ok
check("w8a8_plan_stale", stale_ok)

# ---------------- W4A16 ---------------------------------------------------------
N4, K4 = 1000, 1024
w4 = torch.randn(N4, K4, device=dev, dtype=torch.float16)
wp, sc, deq = quantize_weight_int4(w4, group=128)
x4 = torch.randn(300, K4, device=dev, dtype=torch.float16)
max_err4 = 0.0
for BM in (16, 32, 64, 128):
    for M in list(range(1, 65)) + [72, 99, 128, 129, 200, 288, 300]:
        out, _ = w4a16_matmul(x4[:M], wp, sc, BLOCK_M=BM)
        ref = x4[:M].float() @ deq.float().T
        err = ((out.float() - ref).norm() / ref.norm()).item()
        max_err4 = max(max_err4, err)
        check("w4a16", err < 2e-3, f"BM={BM} M={M} err={err}")
results["w4a16_max_rel_fro_err_vs_dequant_ref"] = max_err4
out, _ = w4a16_matmul(x4, wp, sc)
ref16 = x4.float() @ w4.float().T
results["w4a16_vs_fp16_rel_fro_err"] = ((out.float() - ref16).norm() / ref16.norm()).item()
results["w4a16_binaries_compiled"] = n_compiled(_w4a16_kernel)

# ---------------- Paged attention --------------------------------------------------
H, H_KV, D = 32, 8, 128
max_attn_err = 0.0
for B in (1, 3, 8):
    for k in (2, 4, 8):
        ql = k + 1
        ctx = [int(x) for x in torch.randint(ql, 700, (B,))]
        kc, vc, ks, vs, bt, cl = make_paged_kv(B, ctx, H_KV, D, seed=B * 10 + k)
        q = torch.randn(B, ql, H, D, device=dev, dtype=torch.float16)
        ref = paged_attention_ref(q, kc, vc, ks, vs, bt, cl, ql)
        for QPP in (1, ql):
            out, _ = paged_attention(q, kc, vc, ks, vs, bt, cl, ql, QPP=QPP)
            err = (out.float() - ref.float()).abs().max().item()
            max_attn_err = max(max_attn_err, err)
            check("attn", err < 2e-2, f"B={B} k={k} QPP={QPP} err={err}")
results["paged_attn_max_abs_err"] = max_attn_err

# plan-guarded attention: one fixed grid (B_max=8, q_len_max=9), varying active (B, q_len)
BMAX, QLMAX = 8, 9
ctx = [int(x) for x in torch.randint(QLMAX, 700, (BMAX,))]
kc, vc, ks, vs, bt, cl = make_paged_kv(BMAX, ctx, H_KV, D, seed=99)
qfull = torch.randn(BMAX, QLMAX, H, D, device=dev, dtype=torch.float16)
plan = torch.zeros(8, dtype=torch.int32, device=dev)
stale_ok = True
for B_act, ql in ((1, 3), (3, 5), (8, 9), (5, 3), (8, 5)):
    plan[0], plan[1] = B_act, ql
    out = torch.full_like(qfull, 7.0)
    # q for the active window sits in the first ql slots
    paged_attention(qfull, kc, vc, ks, vs, bt, cl, QLMAX, QPP=QLMAX, plan=plan, out=out)
    ref = paged_attention_ref(qfull[:B_act], kc, vc, ks, vs, bt[:B_act], cl[:B_act], ql)
    err = (out[:B_act, :ql].float() - ref[:, :ql].float()).abs().max().item()
    check("attn_plan", err < 2e-2, f"B={B_act} ql={ql} err={err}")
    stale_ok &= bool((out[B_act:] == 7.0).all().item()) and bool((out[:B_act, ql:] == 7.0).all().item())
results["paged_attn_plan_inactive_untouched"] = stale_ok
check("attn_plan_stale", stale_ok)

results["failures"] = fails
print(json.dumps(results, indent=2))
json.dump(results, open("results/correctness.json", "w"), indent=2)
sys.exit(1 if fails else 0)
