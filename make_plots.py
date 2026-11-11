"""Turn results/*.json into figures + summary tables (results/summary.md)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .3})
md = []

# ---------------- Fig 1: staircase -----------------------------------------------------
mm = json.load(open("results/matmul.json"))
A = pd.DataFrame(mm["partA"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, kind in enumerate(("w8a8", "w4a16")):
    for BM in (16, 32, 64, 128):
        ax[i].plot(A.M, A[f"{kind}_BM{BM}"] * 1e3, label=f"BLOCK_M={BM}", lw=1.2)
    ax[i].plot(A.M, A.fp16_cublas * 1e3, "k--", label="FP16 cuBLAS", lw=1)
    for k in (2, 4, 8):
        for B in (1, 2, 4, 8, 16):
            M = B * (k + 1)
            if M <= A.M.max():
                ax[i].axvline(M, color={2: "C0", 4: "C1", 8: "C2"}[k], alpha=.12, lw=.8)
    ax[i].set_title(f"{kind.upper()}  o_proj 4096x4096  (faint lines: M=B(k+1))")
    ax[i].set_xlabel("M = tokens verified per step"); ax[i].set_ylabel("latency (us)")
    ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig1_staircase.png"); plt.close()

# tile utilisation table for spec-decode shapes
rows = []
for k in (2, 4, 8):
    for B in (1, 2, 4, 8, 16, 32):
        M = B * (k + 1)
        rows.append([k, B, M] + [f"{100 * M / (-(-M // bm) * bm):.0f}%" for bm in (16, 32, 64, 128)])
md.append("## Tile (row) utilisation of BLOCK_M for M = B(k+1)\n")
md.append(pd.DataFrame(rows, columns=["k", "B", "M", "BM16", "BM32", "BM64", "BM128"]).to_markdown(index=False))

# flat region: latency at M=1 vs M=16 vs 17
def at(M, col):
    return float(A.loc[A.M == M, col].iloc[0]) * 1e3
md.append("\n\n## Staircase (o_proj 4096x4096), latency in us\n")
t = []
for col in ("w8a8_BM16", "w8a8_BM64", "w4a16_BM16", "w4a16_BM64", "fp16_cublas"):
    t.append([col] + [f"{at(M, col):.1f}" for M in (1, 3, 9, 16, 17, 27, 32, 33, 45, 64, 72, 80, 144)])
md.append(pd.DataFrame(t, columns=["kernel"] + [f"M={m}" for m in (1, 3, 9, 16, 17, 27, 32, 33, 45, 64, 72, 80, 144)]).to_markdown(index=False))

# ---------------- Fig 2: (B,k) grid -----------------------------------------------------
Bdf = pd.DataFrame(mm["partB"])
Bdf["t_us"] = Bdf.t_best_ms * 1e3
layer_sum = Bdf.groupby(["kind", "B", "k", "M"]).t_us.sum().reset_index()   # full decoder layer linears
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for i, kind in enumerate(("w8a8", "w4a16", "fp16_cublas")):
    d = layer_sum[layer_sum.kind == kind]
    for k in (0, 2, 4, 8):
        dd = d[d.k == k]
        ax[i].plot(dd.B, dd.t_us, "o-", label=f"k={k}" if k else "k=0 (decode)")
    ax[i].set_xscale("log", base=2); ax[i].set_title(f"{kind}: sum of 4 linears / layer (best cfg)")
    ax[i].set_xlabel("batch B"); ax[i].set_ylabel("us"); ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig2_batch_k_grid.png"); plt.close()

piv = layer_sum.pivot_table(index=["B"], columns=["kind", "k"], values="t_us").round(0)
md.append("\n\n## Per-layer linear time (us, best config, Llama-3-8B shapes)\n")
md.append(piv.to_markdown())

# cost of verification relative to plain decode: t(B,k)/t(B,0)
rel = []
for kind in ("w8a8", "w4a16", "fp16_cublas"):
    for B in (1, 2, 4, 8, 16, 32):
        base = layer_sum[(layer_sum.kind == kind) & (layer_sum.B == B) & (layer_sum.k == 0)].t_us.iloc[0]
        rel.append([kind, B] + [f"{layer_sum[(layer_sum.kind == kind) & (layer_sum.B == B) & (layer_sum.k == k)].t_us.iloc[0] / base:.2f}x" for k in (2, 4, 8)])
md.append("\n\n## Verification cost relative to plain decode (same B): t(B,k)/t(B,0)\n")
md.append(pd.DataFrame(rel, columns=["kernel", "B", "k=2", "k=4", "k=8"]).to_markdown(index=False))

# best config chosen per M -> shows tile choice flips with k
bc = Bdf[(Bdf.kind != "fp16_cublas") & (Bdf.layer == "gate_up")][["kind", "B", "k", "M", "best_cfg", "t_us", "t_BM16_ms", "t_BM64_ms"]].copy()
bc["BM16_slowdown"] = (bc.t_BM16_ms * 1e3 / bc.t_us).round(2)
bc["BM64_slowdown"] = (bc.t_BM64_ms * 1e3 / bc.t_us).round(2)
md.append("\n\n## Oracle tile config per shape (gate_up 28672x4096) vs a fixed config\n")
md.append(bc.drop(columns=["t_BM16_ms", "t_BM64_ms"]).round(1).to_markdown(index=False))

# MaxPad cost on linears: run at k=8 when real k is 2 or 4
mp = []
for kind in ("w8a8", "w4a16"):
    for B in (1, 2, 4, 8, 16, 32):
        t8 = layer_sum[(layer_sum.kind == kind) & (layer_sum.B == B) & (layer_sum.k == 8)].t_us.iloc[0]
        row = [kind, B]
        for k in (2, 4):
            tk = layer_sum[(layer_sum.kind == kind) & (layer_sum.B == B) & (layer_sum.k == k)].t_us.iloc[0]
            row.append(f"{100 * (t8 / tk - 1):+.0f}%")
        mp.append(row)
md.append("\n\n## Linear-layer time penalty of padding to k_max=8 (MaxPad)\n")
md.append(pd.DataFrame(mp, columns=["kernel", "B", "real k=2", "real k=4"]).to_markdown(index=False))

# ---------------- Fig 3: registers ----------------------------------------------------
rg = json.load(open("results/regs.json"))
R = pd.DataFrame([r for r in rg["rows"] if "error" not in r])
E = pd.DataFrame([r for r in rg["rows"] if "error" in r])
G = R[R.kind != "attn"].copy()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
allrows = pd.DataFrame([r for r in rg["rows"] if r["kind"] != "attn"])
allrows["status"] = np.where(allrows.get("error").notna(), "fail", np.where(allrows.spills.fillna(0) > 0, "spill", "ok"))
st = allrows.groupby(["kind", "BM", "status"]).size().unstack(fill_value=0)
st = st.div(st.sum(1), axis=0) * 100
xl = [f"{k}\nBM{int(b)}" for k, b in st.index]
bottom = np.zeros(len(st))
for s, c in (("ok", "C2"), ("spill", "C1"), ("fail", "C3")):
    if s in st:
        ax[0].bar(range(len(st)), st[s], bottom=bottom, color=c, label={"ok": "fits in registers", "spill": "spills to local mem", "fail": "OutOfResources (smem)"}[s])
        bottom += st[s].values
ax[0].set_xticks(range(len(st)), xl, fontsize=7); ax[0].set_ylabel("% of configs (BN x BK x warps x stages)")
ax[0].legend(fontsize=7); ax[0].set_title("config viability by BLOCK_M")
g = G.groupby(["kind", "BM"]).agg(regs=("regs", "median"), spills=("spills", "max"), occ=("occupancy", "median")).reset_index()
for kind in ("w8a8", "w4a16"):
    gg = g[g.kind == kind]
    ax[1].plot(gg.BM, gg.regs, "o-", label=f"{kind} median regs")
ax[1].axhline(255, color="r", ls="--", lw=.8, label="255 reg limit")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("BLOCK_M"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig3_registers.png"); plt.close()

md.append("\n\n## Register pressure summary (GEMM, all BN/BK/warps/stages configs)\n")
md.append(G.groupby(["kind", "BM"]).agg(n=("regs", "size"), regs_min=("regs", "min"), regs_max=("regs", "max"),
          spilling_cfgs=("spills", lambda s: int((s > 0).sum())), max_spill=("spills", "max"),
          occ_median=("occupancy", "median"), best_us=("t_us", "min")).round(2).to_markdown())
if len(E):
    md.append("\n\nConfigs that failed to compile/launch (mostly shared-memory overflow on 100 KB/SM):\n")
    md.append(E.groupby(["kind", "error"]).size().to_markdown())
top = G.sort_values("t_us").groupby("kind").head(5)
md.append("\n\n## Fastest 5 configs per kernel at M=72\n")
md.append(top[["kind", "BM", "BN", "BK", "warps", "stages", "regs", "spills", "smem", "occupancy", "limiter", "t_us"]].round(2).to_markdown(index=False))
spill = G[G.spills > 0].sort_values("spills", ascending=False).head(10)
if len(spill):
    md.append("\n\n## Worst spilling configs\n")
    md.append(spill[["kind", "BM", "BN", "BK", "warps", "stages", "regs", "spills", "t_us"]].round(1).to_markdown(index=False))
At = R[R.kind == "attn"]
md.append("\n\n## Attention kernel registers vs k (ctx 2048, B=8)\n")
md.append(At[["k", "QPP", "BLOCK_Q", "row_util", "BN", "warps", "regs", "spills", "occupancy", "limiter", "t_us"]].round(2).to_markdown(index=False))

# ---------------- Fig 4: attention ------------------------------------------------------
at = pd.DataFrame(json.load(open("results/attention.json")))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for ctx, a in ((1024, ax[0]), (4096, ax[1])):
    for B in (1, 8, 32):
        d = at[(at.ctx == ctx) & (at.B == B)]
        fused = d[d.QPP == d.q_len]; dec = d[d.QPP == 1]
        a.plot(fused.k, fused.t_us, "o-", label=f"B={B} fused (QPP=k+1)")
        a.plot(dec.k, dec.t_us, "x--", label=f"B={B} per-token (QPP=1)")
    a.set_yscale("log"); a.set_xlabel("draft length k"); a.set_ylabel("us"); a.set_title(f"INT8 paged attn, ctx={ctx}")
    a.legend(fontsize=6)
d = at[(at.ctx == 4096) & (at.B == 8) & (at.QPP == at.q_len)]
ax[2].bar(d.k, d.row_util * 100, color="C2", alpha=.6, label="BLOCK_Q row utilisation %")
ax2 = ax[2].twinx(); ax2.plot(d.k, d.us_per_token, "ko-", label="us per verified token"); ax2.set_ylabel("us / token")
ax[2].set_xlabel("k"); ax[2].set_ylabel("%"); ax[2].set_title("fused kernel, B=8 ctx=4096 (G=4 heads/kv)")
ax[2].legend(loc="lower left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig4_attention.png"); plt.close()
md.append("\n\n## Paged attention (us); fused vs per-token, ctx=4096\n")
pv = at[at.ctx == 4096].assign(variant=lambda x: np.where(x.QPP == 1, "per-token", "fused")).pivot_table(index=["B"], columns=["variant", "k"], values="t_us").round(0)
md.append(pv.to_markdown())
md.append("\n\n## Fused attention: row utilisation and effective KV bandwidth (ctx=4096)\n")
md.append(at[(at.ctx == 4096) & (at.QPP == at.q_len)][["B", "k", "BLOCK_Q", "row_util", "t_us", "us_per_token", "eff_GBps"]].round(2).to_markdown(index=False))

# ---------------- Fig 5: graphs -------------------------------------------------------
gr = json.load(open("results/graphs.json"))
strat = ("eager", "allshapes", "maxpad", "plan")
mix = {s: {m: np.mean([r[s][m] for r in gr["mixed"]]) for m in ("p50_ms", "p99_ms", "mean_ms", "waste_pct", "tok_per_s")} for s in strat}
tab = [[s, gr.get(f"{s}_graphs", "-"), f"{gr.get(f'{s}_build_s', 0):.2f}" if s != "eager" else "-",
        f"{mix[s]['p50_ms']:.2f}", f"{mix[s]['p99_ms']:.2f}", f"{mix[s]['waste_pct']:.0f}%", f"{mix[s]['tok_per_s']:.0f}"] for s in strat]
md.append("\n\n## CUDA Graph strategies, mixed trace (B~U[1,16], k~{2,4,8}), 4 INT8 layers, mean of 3 seeds\n")
md.append(pd.DataFrame(tab, columns=["strategy", "#graphs", "build s", "P50 ms", "P99 ms", "payload waste", "tokens/s"]).to_markdown(index=False))
ft = []
for k, r in gr["fixed_k"].items():
    ft.append([k] + [f"{r[s]['mean_ms']:.2f}" for s in strat] + [f"{r['maxpad']['waste_pct']:.0f}%", f"{100 * (r['plan']['mean_ms'] / r['maxpad']['mean_ms'] - 1):+.1f}%"])
md.append("\n\n## Fixed-k traces: mean step latency (ms)\n")
md.append(pd.DataFrame(ft, columns=["k", *strat, "maxpad waste", "plan vs maxpad"]).to_markdown(index=False))
md.append(f"\n\nplan-graph vs eager max |diff| over all (B,k): {gr['plan_vs_eager_max_abs_diff']}")

g128 = json.load(open("results/graphs_bm128.json"))
mix128 = {s: {m: np.median([r[s][m] for r in g128["mixed"]]) for m in ("p50_ms", "p99_ms", "mean_ms", "waste_pct", "tok_per_s")} for s in strat}
tab = [[s, f"{g128.get(f'{s}_build_s', 0):.2f}" if s != "eager" else "-", f"{mix128[s]['p50_ms']:.2f}", f"{mix128[s]['p99_ms']:.2f}",
        f"{mix128[s]['waste_pct']:.0f}%", f"{mix128[s]['tok_per_s']:.0f}", f"{mix128[s]['tok_per_s'] / mix[s]['tok_per_s']:.2f}x"] for s in strat]
md.append("\n\n## Same experiment with BLOCK_M=128 GEMM tiles (median of 3 seeds)\n")
md.append(pd.DataFrame(tab, columns=["strategy", "build s", "P50 ms", "P99 ms", "payload waste", "tokens/s", "vs BLOCK_M=16"]).to_markdown(index=False))
ft = []
for k, r in g128["fixed_k"].items():
    ft.append([k] + [f"{r[s]['mean_ms']:.2f}" for s in strat] + [f"{r['maxpad']['waste_pct']:.0f}%", f"{100 * (r['plan']['mean_ms'] / r['maxpad']['mean_ms'] - 1):+.1f}%"])
md.append("\n\n## BLOCK_M=128, fixed-k traces: mean step latency (ms)\n")
md.append(pd.DataFrame(ft, columns=["k", *strat, "maxpad waste", "plan vs maxpad"]).to_markdown(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(len(strat))
ax[0].bar(x - .2, [mix[s]["tok_per_s"] for s in strat], .4, label="GEMM BLOCK_M=16")
ax[0].bar(x + .2, [mix128[s]["tok_per_s"] for s in strat], .4, label="GEMM BLOCK_M=128")
ax[0].set_xticks(x, strat); ax[0].set_ylabel("verified tokens / s"); ax[0].legend(fontsize=7)
ax[0].set_title("mixed (B,k) trace, 4 INT8 layers")
ks = list(gr["fixed_k"].keys())
for i, s in enumerate(strat):
    ax[1].bar(np.arange(len(ks)) + (i - 1.5) * .2, [gr["fixed_k"][k][s]["mean_ms"] for k in ks], .2, label=s)
ax[1].set_xticks(range(len(ks)), [f"k={k}" for k in ks]); ax[1].set_ylabel("mean ms / step"); ax[1].legend(fontsize=7)
ax[1].set_title("fixed draft length (BLOCK_M=16)")
plt.tight_layout(); plt.savefig("figures/fig5_graphs.png"); plt.close()

cj = json.load(open("results/correctness.json"))
md.insert(0, "## Correctness\n\n" + pd.DataFrame(list(cj.items()), columns=["check", "value"]).to_markdown(index=False) + "\n\n")
md.insert(0, f"# Raw results summary - {mm['device']}\n\n")
open("results/summary.md", "w", encoding="utf-8").write("\n".join(md))
print("ok")
