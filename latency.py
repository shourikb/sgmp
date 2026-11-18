import argparse, copy, platform, statistics, time

import torch

from common import append_jsonl, get_loaders, load_fp32


def bench(model, threads, iters=200, warmup=30):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def int8_fx(model, calib_loader, n_batches=4):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = "x86"
    qmap = get_default_qconfig_mapping("x86")
    ex = (torch.randn(1, 3, 224, 224),)
    prepared = prepare_fx(copy.deepcopy(model).eval(), qmap, ex)
    with torch.inference_mode():
        for i, (x, _) in enumerate(calib_loader):
            if i >= n_batches:
                break
            prepared(x)
    return convert_fx(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", default="runs/latency.jsonl")
    args = ap.parse_args()
    L = get_loaders(args.data, 64, workers=4)
    fp32 = load_fp32(args.data, args.ckpt)
    int8 = int8_fx(fp32, L["calib"])
    for t in args.threads:
        row = dict(cpu=platform.processor() or platform.machine(), threads=t,
                   fp32_ms=bench(fp32, t), int8_ms=bench(int8, t))
        row["speedup"] = row["fp32_ms"] / row["int8_ms"]
        print(row)
        append_jsonl(args.out, row)


if __name__ == "__main__":
    main()
