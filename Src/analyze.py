import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False,
})


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def short(n):
    return (n.replace("layer", "L").replace(".conv", ".c").replace(".downsample.0", ".ds"))


def fig_allocations(runs, out, fp32):
    rows = jl(os.path.join(runs, "sgmp", "allocations.jsonl"))
    if not rows:
        return
    budgets = [u for u in sorted({r["u"] for r in rows}, reverse=True)
               if len({tuple(r["bitmap"].values()) for r in rows if r["u"] == u}) > 1]
    order = [("sgmp_greedy", "SGMP (greedy)"), ("sgmp_dp", "SGMP (exact DP)"),
             ("uniform", "Uniform"), ("wmse_greedy", "Weight-MSE heuristic"), ("random", "Random (×10)")]
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 2.4), sharey=True)
    axes = [axes] if len(budgets) == 1 else axes
    for ax, u in zip(axes, budgets):
        R = [r for r in rows if r["u"] == u]
        for y, (key, label) in enumerate(order):
            pts = [r for r in R if (r["algo"].startswith("random") if key == "random" else r["algo"] == key)]
            if not pts:
                continue
            col = C[0] if key.startswith("sgmp") else MUTED
            xs = [p["val_top1"] for p in pts]
            ax.scatter(xs, [y] * len(xs), s=36 if key != "random" else 26, color=col if key != "random" else "none",
                       edgecolor=col, linewidth=1.2, zorder=3)
            if key != "random":
                left = xs[0] > 45
                ax.annotate(f"{xs[0]:.1f}%  ({pts[0]['size_mb']:.2f} MB)", (xs[0], y), xytext=(-8 if left else 8, 0),
                            textcoords="offset points", fontsize=7.5, color=INK2, ha="right" if left else "left", va="center")
            else:
                m = sum(xs) / len(xs)
                ax.annotate(f"mean {m:.1f}%", (m, y), xytext=(0, -9), textcoords="offset points",
                            fontsize=7.5, color=INK2, ha="center", va="top")
        if fp32:
            ax.axvline(fp32, color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
            ax.annotate(f"FP32 {fp32:.1f}%", (fp32, -0.55), xytext=(3, 0), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=INK2)
        ax.set_title(f"Budget = uniform {u:g}-bit size ({R[0]['budget_mb']:.2f} MB)", loc="left")
        ax.set_xlabel("Test top-1 (%), PTQ only, no training")
        ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
        ax.set_ylim(len(order) - 0.2, -0.6)
    axes[0].set_yticks(range(len(order)), [l for _, l in order])
    save(fig, out, "fig_allocations")

    print("\n| Budget | Allocation | Size (MB) | Test top-1 | Test ECE |\n|---|---|---|---|---|")
    for u in sorted({r["u"] for r in rows}, reverse=True):
        R = [r for r in rows if r["u"] == u]
        for r in R:
            if not r["algo"].startswith("random"):
                print(f"| u={u:g} | {r['algo']} | {r['size_mb']:.2f} | {r['val_top1']:.2f} | {r['val_ece']:.2f} |")
        rv = [r["val_top1"] for r in R if r["algo"].startswith("random")]
        rs = [r["size_mb"] for r in R if r["algo"].startswith("random")]
        if rv:
            print(f"| u={u:g} | random ×{len(rv)} (mean [min, max]) | {min(rs):.2f}–{max(rs):.2f} | "
                  f"{sum(rv)/len(rv):.2f} [{min(rv):.2f}, {max(rv):.2f}] | |")


def fig_sensitivity(runs, out, bits=4, u=4):
    p = os.path.join(runs, "sgmp", "sensitivity.json")
    if not os.path.exists(p):
        return
    s = json.load(open(p))
    names = list(s["omega"])
    om = [s["omega"][n][str(bits)] for n in names]
    oh = [s["omega_holdout"][n][str(bits)] for n in names]
    bm = next((r["bitmap"] for r in jl(os.path.join(runs, "sgmp", "allocations.jsonl"))
               if r["u"] == u and r["algo"] == "sgmp_greedy"), None)
    x = list(range(len(names)))
    fig, axes = plt.subplots(2 if bm else 1, 1, figsize=(7, 3.8 if bm else 2.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]} if bm else None)
    ax = axes[0] if bm else axes
    ax.bar(x, om, width=0.6, color=C[0], label="512 calibration images (used by SGMP)", zorder=3)
    ax.scatter(x, oh, s=22, color=C[1], edgecolor=SURF, linewidth=1, zorder=4,
               label="1,000 held-out images (check)")
    rho = s["spearman_calib_vs_holdout"].get(str(bits))
    ax.set_ylabel(f"Ω$_\\ell$({bits}) = Δ CE (nats)")
    ax.set_title(f"Per-layer sensitivity at {bits}-bit weights (Spearman ρ calib vs held-out = {rho:.2f})",
                 loc="left")
    ax.axhline(0, color=AXIS, lw=0.8); ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.set_ylim(top=max(max(om), max(oh)) * 1.35)
    if bm:
        b = [bm[n] for n in names]
        axes[1].bar(x, b, width=0.6, color=C[0], zorder=3)
        for xi, bi in zip(x, b):
            axes[1].annotate(str(bi), (xi, bi), xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=7, color=INK2)
        axes[1].set_ylim(0, 9.5); axes[1].set_yticks([3, 4, 6, 8])
        axes[1].set_ylabel("bits"); axes[1].axhline(u, color=INK2, lw=1, ls=(0, (3, 2)))
        axes[1].set_title(f"SGMP bit allocation at the uniform-{u}-bit size (dashed = uniform)", loc="left")
        axes[1].grid(axis="x", visible=False)
        ax2 = axes[1]
    else:
        ax2 = ax
    counts = s["counts"]
    ax2.set_xticks(x, [f"{short(n)}\n{counts[n]/1e6:.2f}M" if counts[n] >= 1e5 else f"{short(n)}\n{counts[n]/1e3:.0f}k"
                       for n in names], rotation=90, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_sensitivity")

def fig_ptq(runs, out, fp32):
    rows = jl(os.path.join(runs, "ptq", "ptq.jsonl"))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, (meth, lab) in enumerate([("minmax", "Naive min-max"), ("mse", "MSE-optimal scale")]):
        R = sorted([r for r in rows if r["method"] == meth and r["abits"] == 8 and r["wbits"] < 32],
                   key=lambda r: r["size_mb"])
        ax.plot([r["size_mb"] for r in R], [r["top1"] for r in R], "-o", color=C[i], lw=2, ms=5, label=lab)
        for r in R:
            ax.annotate(f"W{r['wbits']}", (r["size_mb"], r["top1"]), xytext=(4, -9 if i else 4),
                        textcoords="offset points", fontsize=7, color=INK2)
    if fp32:
        ax.axhline(fp32, color=INK2, lw=1, ls=(0, (3, 2)))
        ax.annotate("FP32", (ax.get_xlim()[0], fp32), xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("Model size (MB)"); ax.set_ylabel("Test top-1 (%)")
    ax.set_title("Uniform PTQ, 8-bit activations", loc="left"); ax.legend(fontsize=7.5)
    save(fig, out, "fig_ptq")
    print("\n| Method | W/A bits | Size (MB) | Test top-1 | Test ECE |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['method']} | W{r['wbits']}A{r['abits']} | {r['size_mb']:.2f} | {r['top1']:.2f} | {r['ece']:.2f} |")

def fig_qat(runs, out):
    rundirs = sorted(glob.glob(os.path.join(runs, "qat", "*", "log.jsonl")))
    if not rundirs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, p in enumerate(rundirs[:4]):
        R = [r for r in jl(p) if r.get("split") == "holdout"]
        name = os.path.basename(os.path.dirname(p))
        ax.plot([r["epoch"] for r in R], [r["top1"] for r in R], "-o", color=C[i], lw=2, ms=4, label=name)
    ax.set_xlabel("QAT epoch (−1 = PTQ initialisation)"); ax.set_ylabel("Held-out top-1 (%)")
    ax.set_title("QAT training curves", loc="left"); ax.legend(fontsize=7.5)
    save(fig, out, "fig_qat")


def fig_teaser(rows, out, fp32, ptq_rows=None, ymin=60, ymax=90, budgets=(4, 5, 6), qat=None):
    if not rows:
        return
    sg = sorted((float(r["u"]), r["val_top1"]) for r in rows if r["algo"] == "sgmp_greedy")
    un = {float(r["u"]): r["val_top1"] for r in rows if r["algo"] == "uniform"}
    for r in ptq_rows or []:
        if r["method"] == "mse" and r["abits"] == 8 and r["wbits"] < 32 and float(r["wbits"]) not in un:
            un[float(r["wbits"])] = r["top1"]
    un = sorted(p for p in un.items() if p[1] >= ymin)
    sg = [p for p in sg if p[0] in budgets]
    if 8.0 in dict(un):
        sg.append((8.0, dict(un)[8.0]))
    if not un or not sg:
        print("fig_teaser: skipped, need both uniform and sgmp_greedy points")
        return
    if qat:
        fig, (ax, axq) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.32})
    else:
        fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.plot(*zip(*un), marker="s", color=C[1], lw=2, ms=6, mec=SURF, mew=1.2, label="Uniform precision", zorder=3)
    ax.plot(*zip(*sg), marker="o", color=C[0], lw=2, ms=6, mec=SURF, mew=1.2, label="SGMP (ours)", zorder=4)
    if fp32:
        ax.axhline(fp32, color=INK2, lw=1, ls=(0, (3, 2)), zorder=2, label="Full precision (FP32)")
    u4 = min(un, key=lambda p: abs(p[0] - 4)); s4 = min(sg, key=lambda p: abs(p[0] - u4[0]))
    ax.annotate("", xy=(u4[0] + 0.07, s4[1] - 0.6), xytext=(u4[0] + 0.07, u4[1] + 0.6),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.annotate(f"+{s4[1]-u4[1]:.1f} points", (u4[0] + 0.13, u4[1] + 0.6 * (s4[1] - u4[1])),
                fontsize=7.5, color=INK, va="center", ha="left")
    xs = sorted({p[0] for p in un + sg})
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25); ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.set_ylim(ymin, ymax); ax.set_yticks(range(ymin, ymax + 1, 10))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Bits per weight")
    ax.set_ylabel("Test top-1 (%)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.3)
    ax.set_title("Quantization only (PTQ)", loc="left", fontsize=8.5, pad=22)
    if qat:
        items = [("SGMP", s4[1], qat["sgmp"], C[0], "o"), ("Uniform", u4[1], qat["uniform"], C[1], "s")]
        for y, (lab, before, after, col, mk) in enumerate(items):
            axq.plot([before, after], [y, y], color=col, lw=2, zorder=2)
            axq.plot([before], [y], marker=mk, ms=6.5, color=col, mec=SURF, mew=1.2, ls="none", zorder=3)
            axq.plot([after], [y], marker=mk, ms=6.5, mfc=SURF, mec=col, mew=1.6, ls="none", zorder=3)
            axq.annotate(f"{before:.1f}", (before, y), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=7.5, color=INK2)
            axq.annotate(f"{after:.1f}", (after, y), xytext=(0, -9), textcoords="offset points",
                         ha="center", va="top", fontsize=7.5, color=INK,
                         bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"), zorder=6)
        if fp32:
            axq.axvline(fp32, color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
        axq.set_yticks(range(len(items)), [i[0] for i in items]); axq.set_ylim(len(items) - 0.4, -0.6)
        lo = min(70, 5 * int(min(v for i in items for v in i[1:3]) // 5))
        axq.set_xlim(lo, ymax); axq.set_xticks(range(lo, ymax + 1, 5))
        axq.set_xlabel("Test top-1 (%)"); axq.grid(axis="y", visible=False)
        axq.tick_params(axis="y", length=0)
        axq.set_title("4 bits: before → after QAT", loc="left", fontsize=8.5, pad=22)
        axq.plot([], [], marker="o", ms=6, color=INK2, mec=SURF, ls="none", label="PTQ")
        axq.plot([], [], marker="o", ms=6, mfc=SURF, mec=INK2, mew=1.4, ls="none", label="QAT, 10 epochs")
        axq.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
                   handlelength=1.0, columnspacing=1.0, borderaxespad=0.3)
    save(fig, out, "fig_teaser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--qat_uniform", default="u4_s0", help="run name under <runs>/qat for uniform 4-bit QAT")
    ap.add_argument("--qat_sgmp", default="sgmp4_s0", help="run name under <runs>/qat for SGMP 4-bit QAT")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    f = os.path.join(args.runs, "fp32", "final.json")
    fp32 = json.load(open(f))["top1"] if os.path.exists(f) else None
    qat = {}
    for key, d in (("uniform", args.qat_uniform), ("sgmp", args.qat_sgmp)):
        f = os.path.join(args.runs, "qat", d, "final.json")
        if os.path.exists(f):
            qat[key] = json.load(open(f))["top1"]
    fig_teaser(jl(os.path.join(args.runs, 'sgmp', 'allocations.jsonl')), args.out, fp32,
               jl(os.path.join(args.runs, 'ptq', 'ptq.jsonl')), qat=qat if len(qat) == 2 else None)
    fig_allocations(args.runs, args.out, fp32)
    fig_sensitivity(args.runs, args.out)
    fig_ptq(args.runs, args.out, fp32)
    fig_qat(args.runs, args.out)


if __name__ == "__main__":
    main()