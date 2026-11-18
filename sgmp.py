import argparse, copy, json, os, random

import numpy as np
import torch
import torch.nn.functional as F

from common import append_jsonl, device, evaluate, get_loaders, load_fp32
from quant import (FP, calibrate_activations, init_weight_scales, layer_param_counts, qrange,
                   quantizable_layers, quantize_model, scale_mse, set_wbits)

LADDER = [8, 6, 4, 3]

@torch.no_grad()
def mean_ce(model, batches, dev):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        tot += F.cross_entropy(model(x.to(dev)), y.to(dev), reduction="sum").item(); n += len(y)
    return tot / n


def profile(base, loader, dev, calib_loader):
    names = quantizable_layers(base)
    middle = names[1:-1]
    m = quantize_model(copy.deepcopy(base), {n: FP for n in middle}, abits=8).to(dev)
    for n in (names[0], names[-1]):
        set_wbits(m, n, 8)
    calibrate_activations(m, calib_loader, dev, "mse")
    batches = list(loader)
    L0 = mean_ce(m, batches, dev)
    omega = {}
    for n in middle:
        omega[n] = {}
        for b in LADDER:
            set_wbits(m, n, b)
            omega[n][b] = mean_ce(m, batches, dev) - L0
        set_wbits(m, n, FP)
        print(n, {b: round(v, 4) for b, v in omega[n].items()}, flush=True)
    return L0, omega


@torch.no_grad()
def weight_mse_proxy(base, middle):
    out = {}
    for n in middle:
        w = base.get_submodule(n).weight.float()
        out[n] = {}
        for b in LADDER:
            qn, qp = qrange(b, True)
            s = scale_mse(w, b, True)
            out[n][b] = (torch.clamp((w / s).round(), -qn, qp) * s - w).pow(2).sum().item()
    return out


def monotone(omega):
    out = {}
    for n, d in omega.items():
        prev, out[n] = 0.0, {}
        for b in LADDER:
            prev = max(prev, d[b]); out[n][b] = prev
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def size_bits(counts, bitmap, edges, other):
    return sum((8 if n in edges else bitmap[n]) * c for n, c in counts.items()) + 32 * other


def greedy(omega, counts, edges, other, budget_bits):
    bm = {n: 8 for n in omega}
    while size_bits(counts, bm, edges, other) > budget_bits:
        best = None
        for n, b in bm.items():
            i = LADDER.index(b)
            if i + 1 == len(LADDER):
                continue
            b2 = LADDER[i + 1]
            score = (omega[n][b2] - omega[n][b]) / ((b - b2) * counts[n])
            if best is None or score < best[0]:
                best = (score, n, b2)
        if best is None:
            raise ValueError("budget below the all-3-bit size")
        bm[best[1]] = best[2]
    return top_up(bm, counts, edges, other, budget_bits, omega=omega)


def dp(omega, counts, edges, other, budget_bits, unit=4096):
    layers = list(omega)
    fixed = sum(8 * counts[n] for n in edges) + 32 * other
    cap = int((budget_bits - fixed) // unit)
    INF = float("inf")
    best = [INF] * (cap + 1); best[0] = 0.0
    choice = []
    for n in layers:
        new, ch = [INF] * (cap + 1), [None] * (cap + 1)
        for b in LADDER:
            w = -(-b * counts[n] // unit)
            for c in range(w, cap + 1):
                v = best[c - w] + omega[n][b]
                if v < new[c]:
                    new[c], ch[c] = v, b
        best = new; choice.append(ch)
    c = min(range(cap + 1), key=lambda k: best[k])
    if best[c] == INF:
        raise ValueError("infeasible budget")
    bm = {}
    for n, ch in zip(reversed(layers), reversed(choice)):
        b = ch[c]; bm[n] = b; c -= -(-b * counts[n] // unit)
    return bm


def top_up(bm, counts, edges, other, budget_bits, omega=None, rng=None):
    size = size_bits(counts, bm, edges, other)
    while True:
        cands = []
        for n, b in bm.items():
            i = LADDER.index(b)
            if i > 0 and size + (LADDER[i - 1] - b) * counts[n] <= budget_bits:
                cands.append((n, LADDER[i - 1]))
        if not cands:
            return bm
        if omega is None:
            n, b2 = rng.choice(cands)
        else:
            n, b2 = max(cands, key=lambda c: (omega[c[0]][bm[c[0]]] - omega[c[0]][c[1]]) / ((c[1] - bm[c[0]]) * counts[c[0]]))
        size += (b2 - bm[n]) * counts[n]; bm[n] = b2


def random_alloc(middle, counts, edges, other, budget_bits, rng):
    bm = {n: 8 for n in middle}
    while size_bits(counts, bm, edges, other) > budget_bits:
        cands = [k for k, b in bm.items() if b != LADDER[-1]]
        if not cands:
            raise ValueError("budget below the all-3-bit size")
        n = rng.choice(cands)
        bm[n] = LADDER[LADDER.index(bm[n]) + 1]
    return top_up(bm, counts, edges, other, budget_bits, rng=rng)

def ptq_score(base, bm, L, dev):
    m = quantize_model(copy.deepcopy(base), bm, abits=8).to(dev)
    init_weight_scales(m, "mse")
    calibrate_activations(m, L["calib"], dev, "mse")
    h, v = evaluate(m, L["holdout"], dev), evaluate(m, L["val"], dev)
    return {"holdout_top1": h["top1"], "val_top1": v["top1"], "val_ece": v["ece"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/sgmp")
    ap.add_argument("--budgets", default="6,5,4,3.5", help="uniform-equivalent bits (may be fractional)")
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = device(); os.makedirs(args.out, exist_ok=True)
    L = get_loaders(args.data, 256, args.workers)
    base = load_fp32(args.data, args.ckpt, fold=True)

    names = quantizable_layers(base)
    middle = names[1:-1]
    edges = {names[0], names[-1]}
    counts = layer_param_counts(base)
    weight_ids = {id(base.get_submodule(n).weight) for n in names}
    other = sum(p.numel() for p in base.parameters() if id(p) not in weight_ids)

    L0, omega = profile(base, L["calib"], dev, L["calib"])
    _, omega_ho = profile(base, L["holdout"], dev, L["calib"])
    wmse = weight_mse_proxy(base, middle)
    rho = {b: spearman([omega[n][b] for n in middle], [omega_ho[n][b] for n in middle]) for b in (4, 3)}
    rho_w = {b: spearman([omega[n][b] for n in middle], [wmse[n][b] for n in middle]) for b in (4, 3)}
    print("Spearman Omega(calib) vs Omega(holdout):", rho, "| Omega vs weight-MSE:", rho_w)
    json.dump({"L0": L0, "omega": omega, "omega_holdout": omega_ho, "weight_mse": wmse,
               "spearman_calib_vs_holdout": rho, "spearman_omega_vs_wmse": rho_w, "counts": counts},
              open(os.path.join(args.out, "sensitivity.json"), "w"), indent=1)

    omega = monotone(omega)
    rng = random.Random(0)
    table = os.path.join(args.out, "allocations.jsonl")
    for u in map(float, args.budgets.split(",")):
        B = size_bits(counts, {n: u for n in middle}, edges, other)
        allocs = {"uniform": {n: int(u) for n in middle}} if u in LADDER else {}
        allocs.update({
                  "sgmp_greedy": greedy(omega, counts, edges, other, B),
                  "sgmp_dp": dp(omega, counts, edges, other, B),
                  "wmse_greedy": greedy(wmse, counts, edges, other, B)})
        for k in range(args.n_random):
            allocs[f"random{k}"] = random_alloc(middle, counts, edges, other, B, rng)
        for algo, bm in allocs.items():
            info = dict(u=u, algo=algo, budget_mb=B / 8e6,
                        size_mb=size_bits(counts, bm, edges, other) / 8e6,
                        sum_omega=sum(omega[n][bm[n]] for n in middle))
            info.update(ptq_score(base, bm, L, dev))
            print(info, "bits:", list(bm.values()), flush=True)
            append_jsonl(table, dict(info, bitmap=bm))
            json.dump(dict(info, bitmap=bm), open(os.path.join(args.out, f"bitmap_{algo}_u{u:g}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
