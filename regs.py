"""Register pressure / spills / shared memory / occupancy per tile config.

Reads n_regs, n_spills and shared memory from Triton's compiled kernel and
computes theoretical occupancy for the device (sm_86 limits by default).
"""
import json, math
import torch, triton
from triton.testing import do_bench
from kernels import *

dev = "cuda"
props = torch.cuda.get_device_properties(0)
REGS_SM = 65536
MAX_WARPS_SM = props.max_threads_per_multi_processor // 32
SMEM_SM = getattr(props, "shared_memory_per_multiprocessor", 102400)
MAX_BLOCKS_SM = 16


def occupancy(n_regs, num_warps, smem):
    regs_per_warp = math.ceil(n_regs * 32 / 256) * 256
    by_regs = REGS_SM // (regs_per_warp * num_warps) if n_regs else MAX_BLOCKS_SM
    by_warps = MAX_WARPS_SM // num_warps
    by_smem = SMEM_SM // (smem + 1024) if smem else MAX_BLOCKS_SM   # +1KB reserved/block
    blocks = min(by_regs, by_warps, by_smem, MAX_BLOCKS_SM)
    limiter = min((("regs", by_regs), ("warps", by_warps), ("smem", by_smem)), key=lambda x: x[1])[0]
    return blocks, blocks * num_warps / MAX_WARPS_SM, limiter


def meta(ck):
    md = ck.metadata
    smem = md.shared if hasattr(md, "shared") else md["shared"]
    return ck.n_regs, ck.n_spills, smem


rows = []
N, K = 4096, 4096
w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
w_q, sw = quantize_weight_int8(w)
wp, sc, _ = quantize_weight_int4(w, 128)
M = 72                                # B=8, k=8
x = torch.randn(M, K, device=dev, dtype=torch.float16)
a_q, sa = quantize_act_int8(x)

for kind in ("w8a8", "w4a16"):
    for BM in (16, 32, 64, 128):
        for BN in (64, 128, 256):
            for BK in (64, 128):
                for nw in (4, 8):
                    for ns in (2, 3, 4):
                        try:
                            if kind == "w8a8":
                                f = lambda: w8a8_matmul(a_q, sa, w_q, sw, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                                        num_warps=nw, num_stages=ns)
                            else:
                                f = lambda: w4a16_matmul(x, wp, sc, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                                         num_warps=nw, num_stages=ns)
                            _, ck = f()
                            torch.cuda.synchronize()
                            n_regs, n_spills, smem = meta(ck)
                            t = do_bench(f, warmup=5, rep=25, return_mode="median")
                        except Exception as e:
                            rows.append(dict(kind=kind, BM=BM, BN=BN, BK=BK, warps=nw, stages=ns,
                                             error=type(e).__name__))
                            continue
                        blocks, occ, lim = occupancy(n_regs, nw, smem)
                        rows.append(dict(kind=kind, BM=BM, BN=BN, BK=BK, warps=nw, stages=ns,
                                         regs=n_regs, spills=n_spills, smem=smem,
                                         blocks_per_sm=blocks, occupancy=occ, limiter=lim,
                                         t_us=t * 1000))
                        print(rows[-1], flush=True)

# attention: BLOCK_Q x BLOCK_N
H, H_KV, D = 32, 8, 128
B = 8
ctx = [2048] * B
kc, vc, ks, vs, bt, cl = make_paged_kv(B, ctx, H_KV, D)
for k in (1, 2, 3, 4, 7, 8):
    ql = k + 1
    q = torch.randn(B, ql, H, D, device=dev, dtype=torch.float16)
    for QPP in sorted({1, ql}):
        for BN in (16, 32, 64, 128):
            for nw in (4, 8):
                try:
                    f = lambda: paged_attention(q, kc, vc, ks, vs, bt, cl, ql, QPP=QPP, BLOCK_N=BN, num_warps=nw)
                    _, ck = f()
                    torch.cuda.synchronize()
                    n_regs, n_spills, smem = meta(ck)
                    t = do_bench(f, warmup=5, rep=25, return_mode="median")
                except Exception as e:
                    rows.append(dict(kind="attn", k=k, QPP=QPP, BN=BN, warps=nw, error=type(e).__name__))
                    continue
                BQ = max(16, triton.next_power_of_2(QPP * H // H_KV))
                blocks, occ, lim = occupancy(n_regs, nw, smem)
                rows.append(dict(kind="attn", k=k, QPP=QPP, BLOCK_Q=BQ, BN=BN, warps=nw, regs=n_regs,
                                 spills=n_spills, smem=smem, blocks_per_sm=blocks, occupancy=occ,
                                 limiter=lim, t_us=t * 1000, row_util=QPP * H // H_KV / BQ))
                print(rows[-1], flush=True)

json.dump(dict(device=props.name, sms=props.multi_processor_count, rows=rows),
          open("results/regs.json", "w"), indent=1)
