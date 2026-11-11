"""
INT8 / INT4 Triton kernels for speculative-decoding verification.

  * w8a8_matmul    : INT8 activations x INT8 weights -> INT32 acc (tensor cores),
                     per-token activation scale, per-channel weight scale.
  * w4a16_matmul   : FP16 activations x packed INT4 weights (group-wise scales),
                     dequantised in-register, FP16 tensor-core dot.
  * paged_attention: verification attention over an INT8 paged KV cache
                     (per-token/per-head scales), causal over the k+1 query tokens,
                     GQA-aware. QPP (queries-per-program) selects between a
                     decode-style kernel (QPP=1) and a fused verification
                     kernel (QPP=q_len).

Every kernel has a USE_PLAN switch: when set, the real problem size (M, or
B/q_len) is read from a small GPU-resident "plan" tensor and programs outside
the active extent exit early. This is the PolyGraph-style mechanism that lets a
single CUDA graph, captured at the maximum grid, serve every dynamic shape.
"""
import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# W8A8 INT8 GEMM:  C[M,N] = (A_q[M,K] @ W_q[N,K]^T) * sa[M] * sw[N]
# ----------------------------------------------------------------------------
@triton.jit
def _w8a8_kernel(a_ptr, w_ptr, c_ptr, sa_ptr, sw_ptr, plan_ptr,
                 M, N, K,
                 s_am, s_ak, s_wn, s_wk, s_cm, s_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 USE_PLAN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if USE_PLAN:
        M = tl.load(plan_ptr)
    if pid_m * BLOCK_M >= M:           # plan guard / dead tile
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * s_am + rk[None, :] * s_ak
    w_ptrs = w_ptr + rk[:, None] * s_wk + rn[None, :] * s_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0)
        w = tl.load(w_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * s_ak
        w_ptrs += BLOCK_K * s_wk
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sw = tl.load(sw_ptr + rn, mask=rn < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    c_ptrs = c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c_ptrs, c.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def quantize_act_int8(x):
    """Dynamic per-token symmetric INT8 quantisation."""
    s = x.abs().amax(dim=-1).float().clamp(min=1e-8) / 127.0
    q = torch.round(x.float() / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s


def quantize_weight_int8(w):
    s = w.abs().amax(dim=-1).float().clamp(min=1e-8) / 127.0
    q = torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s


def w8a8_matmul(a_q, sa, w_q, sw, out=None, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128,
                num_warps=4, num_stages=3, plan=None, grid_m_max=None):
    M, K = a_q.shape
    N = w_q.shape[0]
    if out is None:
        out = torch.empty((M, N), device=a_q.device, dtype=torch.float16)
    m_for_grid = grid_m_max if grid_m_max is not None else M
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(m_for_grid, BLOCK_M))
    ck = _w8a8_kernel[grid](
        a_q, w_q, out, sa, sw, plan if plan is not None else sa,
        M, N, K,
        a_q.stride(0), a_q.stride(1), w_q.stride(0), w_q.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        USE_PLAN=plan is not None, num_warps=num_warps, num_stages=num_stages)
    return out, ck


# ----------------------------------------------------------------------------
# W4A16 GEMM.  Packing: byte j of row n holds w[n, j] (low nibble) and
# w[n, j + K/2] (high nibble). Symmetric int4 in [-8, 7] stored with +8 offset.
# Group-wise fp16 scales, group size G along K.
# ----------------------------------------------------------------------------
@triton.jit
def _w4a16_kernel(a_ptr, wp_ptr, sc_ptr, c_ptr, plan_ptr,
                  M, N, K,
                  s_am, s_ak, s_wn, s_wk, s_sn, s_sg, s_cm, s_cn,
                  GROUP: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  USE_PLAN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if USE_PLAN:
        M = tl.load(plan_ptr)
    if pid_m * BLOCK_M >= M:
        return
    HALF = K // 2
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, HALF, BLOCK_K):
        kk = k0 + rk
        # packed weight tile [BLOCK_K, BLOCK_N]
        wp = tl.load(wp_ptr + kk[:, None] * s_wk + rn[None, :] * s_wn,
                     mask=(kk[:, None] < HALF) & (rn[None, :] < N), other=0x88)
        lo = (wp & 0xF).to(tl.int8) - 8
        hi = ((wp >> 4) & 0xF).to(tl.int8) - 8
        # scales for the two halves (one group per BLOCK_K when BLOCK_K <= GROUP)
        g_lo = kk // GROUP
        g_hi = (kk + HALF) // GROUP
        sc_lo = tl.load(sc_ptr + rn[None, :] * s_sn + g_lo[:, None] * s_sg,
                        mask=rn[None, :] < N, other=0.0)
        sc_hi = tl.load(sc_ptr + rn[None, :] * s_sn + g_hi[:, None] * s_sg,
                        mask=rn[None, :] < N, other=0.0)
        w_lo = lo.to(tl.float16) * sc_lo
        w_hi = hi.to(tl.float16) * sc_hi
        a_lo = tl.load(a_ptr + rm[:, None] * s_am + kk[None, :] * s_ak,
                       mask=(rm[:, None] < M) & (kk[None, :] < HALF), other=0.0)
        a_hi = tl.load(a_ptr + rm[:, None] * s_am + (kk[None, :] + HALF) * s_ak,
                       mask=(rm[:, None] < M) & (kk[None, :] < HALF), other=0.0)
        acc += tl.dot(a_lo, w_lo)
        acc += tl.dot(a_hi, w_hi)
    c_ptrs = c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def quantize_weight_int4(w, group=128):
    N, K = w.shape
    assert K % 2 == 0 and (K // 2) % group == 0
    wg = w.float().view(N, K // group, group)
    sc = wg.abs().amax(-1).clamp(min=1e-8) / 7.0                      # [N, K/G]
    q = torch.round(wg / sc[..., None]).clamp(-8, 7).view(N, K).to(torch.int16) + 8
    lo, hi = q[:, : K // 2], q[:, K // 2:]
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()              # [N, K/2]
    deq = ((q - 8).float().view(N, K // group, group) * sc[..., None]).view(N, K)
    return packed, sc.to(torch.float16).contiguous(), deq.to(torch.float16)


def w4a16_matmul(a, wp, sc, group=128, out=None, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                 num_warps=4, num_stages=3, plan=None, grid_m_max=None):
    M, K = a.shape
    N = wp.shape[0]
    assert BLOCK_K <= group and group % BLOCK_K == 0
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    m_for_grid = grid_m_max if grid_m_max is not None else M
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(m_for_grid, BLOCK_M))
    ck = _w4a16_kernel[grid](
        a, wp, sc, out, plan if plan is not None else sc,
        M, N, K,
        a.stride(0), a.stride(1), wp.stride(0), wp.stride(1),
        sc.stride(0), sc.stride(1), out.stride(0), out.stride(1),
        GROUP=group, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        USE_PLAN=plan is not None, num_warps=num_warps, num_stages=num_stages)
    return out, ck


# ----------------------------------------------------------------------------
# Verification paged attention over an INT8 KV cache.
#   q        : [B, QL_MAX, H, D] fp16   (k+1 query tokens per sequence)
#   k/v cache: [num_pages, PAGE, H_KV, D] int8, scales [num_pages, PAGE, H_KV] fp16
#   block_tbl: [B, max_pages] int32, ctx_len: [B] int32 (includes the q tokens)
# Program = (sequence b, kv head, chunk of QPP query tokens).  All G = H/H_KV
# query heads sharing that kv head are packed into the M dimension of the tile:
# rows = QPP * G, padded to BLOCK_Q (>=16, tl.dot minimum).
# ----------------------------------------------------------------------------
@triton.jit
def _paged_attn_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, cl_ptr, o_ptr,
                       plan_ptr, q_len, n_seqs, sm_scale,
                       s_qb, s_qt, s_qh, s_kp, s_ko, s_kh, s_sp, s_so, s_sh,
                       s_bt, s_ob, s_ot, s_oh,
                       G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                       QPP: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
                       USE_PLAN: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    qc = tl.program_id(2)
    if USE_PLAN:
        n_seqs = tl.load(plan_ptr)
        q_len = tl.load(plan_ptr + 1)
    if b >= n_seqs:
        return
    if qc * QPP >= q_len:
        return
    L = tl.load(cl_ptr + b)
    r = tl.arange(0, BLOCK_Q)
    qi = qc * QPP + r // G                 # query-token index within the k+1 window
    head = kvh * G + r % G
    row_ok = (r < QPP * G) & (qi < q_len)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * s_qb + qi[:, None] * s_qt + head[:, None] * s_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    q_pos = L - q_len + qi                 # absolute position of each query token
    m_i = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, D), tl.float32)
    # last key any row in this program can see
    kv_end = L - q_len + tl.minimum(qc * QPP + QPP, q_len)
    tn = tl.arange(0, BLOCK_N)
    for start in range(0, kv_end, BLOCK_N):
        t = start + tn
        t_ok = t < kv_end
        page = tl.load(bt_ptr + b * s_bt + t // PAGE, mask=t_ok, other=0)
        off = t % PAGE
        kv_base = page * s_kp + off * s_ko + kvh * s_kh
        k = tl.load(kc_ptr + kv_base[:, None] + d[None, :], mask=t_ok[:, None], other=0)
        v = tl.load(vc_ptr + kv_base[:, None] + d[None, :], mask=t_ok[:, None], other=0)
        sc_base = page * s_sp + off * s_so + kvh * s_sh
        ks = tl.load(ks_ptr + sc_base, mask=t_ok, other=0.0).to(tl.float32)
        vs = tl.load(vs_ptr + sc_base, mask=t_ok, other=0.0).to(tl.float32)
        # per-token K scale factors out of the dot product -> apply after
        s = tl.dot(q, tl.trans(k.to(tl.float16))).to(tl.float32) * (ks[None, :] * sm_scale)
        causal = (t[None, :] <= q_pos[:, None]) & t_ok[None, :]
        s = tl.where(causal, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        # per-token V scale folded into P
        acc = acc * alpha[:, None] + tl.dot((p * vs[None, :]).to(tl.float16), v.to(tl.float16))
        m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(o_ptr + b * s_ob + qi[:, None] * s_ot + head[:, None] * s_oh + d[None, :],
             out.to(tl.float16), mask=row_ok[:, None])


def paged_attention(q, kc, vc, ks, vs, block_tbl, ctx_len, q_len, QPP=None, BLOCK_N=64,
                    num_warps=4, num_stages=2, plan=None, n_seqs=None, out=None):
    B, QLMAX, H, D = q.shape
    H_KV = kc.shape[2]
    G = H // H_KV
    PAGE = kc.shape[1]
    if QPP is None:
        QPP = q_len
    BLOCK_Q = max(16, triton.next_power_of_2(QPP * G))
    if out is None:
        out = torch.zeros_like(q)
    n_seqs = B if n_seqs is None else n_seqs
    qpp_grid = triton.cdiv(QLMAX if plan is not None else q_len, QPP)
    grid = (B if plan is not None else n_seqs, H_KV, qpp_grid)
    ck = _paged_attn_kernel[grid](
        q, kc, vc, ks, vs, block_tbl, ctx_len, out,
        plan if plan is not None else ctx_len, q_len, n_seqs, 1.0 / (D ** 0.5),
        q.stride(0), q.stride(1), q.stride(2),
        kc.stride(0), kc.stride(1), kc.stride(2),
        ks.stride(0), ks.stride(1), ks.stride(2),
        block_tbl.stride(0), out.stride(0), out.stride(1), out.stride(2),
        G=G, D=D, PAGE=PAGE, QPP=QPP, BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
        USE_PLAN=plan is not None, num_warps=num_warps, num_stages=num_stages)
    return out, ck


def make_paged_kv(B, ctx_len, H_KV, D, PAGE=16, device="cuda", seed=0):
    """Random INT8 KV cache with a shuffled (non-contiguous) page table."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_seq = [(int(l) + PAGE - 1) // PAGE for l in ctx_len]
    total = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    perm = torch.randperm(total, generator=g)
    bt = torch.zeros((B, max_pages), dtype=torch.int32)
    i = 0
    for b, n in enumerate(pages_per_seq):
        bt[b, :n] = perm[i:i + n].to(torch.int32)
        i += n
    kf = torch.randn((total, PAGE, H_KV, D), generator=g).half()
    vf = torch.randn((total, PAGE, H_KV, D), generator=g).half()

    def q8(x):
        s = x.float().abs().amax(-1).clamp(min=1e-8) / 127.0
        return torch.round(x.float() / s[..., None]).clamp(-127, 127).to(torch.int8), s.half()
    kc, ks = q8(kf)
    vc, vs = q8(vf)
    to = lambda t: t.to(device).contiguous()
    return to(kc), to(vc), to(ks), to(vs), to(bt), torch.tensor(ctx_len, dtype=torch.int32, device=device)


def paged_attention_ref(q, kc, vc, ks, vs, block_tbl, ctx_len, q_len):
    B, _, H, D = q.shape
    H_KV = kc.shape[2]
    G = H // H_KV
    PAGE = kc.shape[1]
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(B):
        L = int(ctx_len[b])
        idx = torch.arange(L, device=q.device)
        pg = block_tbl[b, idx // PAGE].long()
        off = idx % PAGE
        k = kc[pg, off].float() * ks[pg, off].float()[..., None]       # [L, H_KV, D]
        v = vc[pg, off].float() * vs[pg, off].float()[..., None]
        k = k.repeat_interleave(G, dim=1)
        v = v.repeat_interleave(G, dim=1)
        qq = q[b, :q_len].float()                                        # [ql, H, D]
        s = torch.einsum("qhd,khd->hqk", qq, k) / D ** 0.5
        qpos = L - q_len + torch.arange(q_len, device=q.device)
        s = s.masked_fill(idx[None, None, :] > qpos[None, :, None], float("-inf"))
        out[b, :q_len] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v)
    return out.half()
