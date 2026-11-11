"""INT8-KV paged verification attention: effect of draft length k and batch B.

Compares the decode-style kernel (QPP=1: one program per query token, KV
re-read k+1 times) with the fused verification kernel (QPP=k+1: all k+1 tokens
x G query heads packed into one M tile, KV read once).
"""
import json
import torch, triton
from triton.testing import do_bench
from kernels import *

dev = "cuda"
H, H_KV, D = 32, 8, 128          # Llama-3-8B attention geometry (GQA group 4)
G = H // H_KV
rows = []
for ctx_len in (1024, 4096):
    for B in (1, 2, 4, 8, 16, 32):
        kc, vc, ks, vs, bt, cl = make_paged_kv(B, [ctx_len] * B, H_KV, D, seed=B)
        kv_bytes = B * ctx_len * H_KV * D * 2 + B * ctx_len * H_KV * 2 * 2   # int8 K,V + fp16 scales
        for k in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            ql = k + 1
            q = torch.randn(B, ql, H, D, device=dev, dtype=torch.float16)
            for QPP in sorted({1, ql}):
                best = None
                for BN in (32, 64):
                    f = lambda: paged_attention(q, kc, vc, ks, vs, bt, cl, ql, QPP=QPP, BLOCK_N=BN)
                    try:
                        f()
                        t = do_bench(f, warmup=5, rep=30, return_mode="median")
                    except Exception:
                        continue
                    if best is None or t < best[0]:
                        best = (t, BN)
                BQ = max(16, triton.next_power_of_2(QPP * G))
                rows.append(dict(ctx=ctx_len, B=B, k=k, q_len=ql, QPP=QPP, BLOCK_Q=BQ,
                                 row_util=QPP * G / BQ, t_us=best[0] * 1000, BN=best[1],
                                 eff_GBps=kv_bytes / (best[0] * 1e-3) / 1e9,
                                 us_per_token=best[0] * 1000 / (B * ql)))
                print(rows[-1], flush=True)
        del kc, vc, ks, vs
        torch.cuda.empty_cache()
json.dump(rows, open("results/attention.json", "w"), indent=1)
