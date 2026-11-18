import argparse, copy, json, os

import torch

from common import append_jsonl, device, evaluate, get_loaders, load_fp32
from quant import FP, calibrate_activations, init_weight_scales, quantize_model, size_mb

CONFIGS = [(FP, FP), (8, 8), (6, 8), (4, 8), (3, 8), (8, 6), (8, 4), (6, 6), (4, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/ptq")
    ap.add_argument("--methods", default="minmax,mse")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = device(); os.makedirs(args.out, exist_ok=True)
    L = get_loaders(args.data, 256, args.workers)

    unf, fol = load_fp32(args.data, args.ckpt), load_fp32(args.data, args.ckpt, fold=True)
    x, _ = next(iter(L["calib"]))
    with torch.no_grad():
        diff = (unf(x[:32]) - fol(x[:32])).abs().max().item()
    print(f"BN-fold max |logit diff| (CPU fp32) = {diff:.2e}"); assert diff < 1e-3, "BN folding is wrong"
    base = fol

    for method in args.methods.split(","):
        for wb, ab in CONFIGS:
            m = quantize_model(copy.deepcopy(base), wb, ab).to(dev)
            init_weight_scales(m, method)
            if ab < FP:
                calibrate_activations(m, L["calib"], dev, method)
            r = evaluate(m, L["val"], dev)
            row = dict(method=method, wbits=wb, abits=ab, size_mb=round(size_mb(m), 3), **r)
            print(row); append_jsonl(os.path.join(args.out, "ptq.jsonl"), row)


if __name__ == "__main__":
    main()
