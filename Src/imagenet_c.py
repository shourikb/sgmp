import argparse, json, os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.data import ImageNetInfo

from common import MEAN, STD, append_jsonl, device, evaluate, get_loaders, load_fp32
from quant import quantize_model

CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
               "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
               "elastic_transform", "pixelate", "jpeg_compression"]


def our_wnids(data_root):
    classes = json.load(open(os.path.join(data_root, "classes.json")))
    wnids = ImageNetInfo().label_names()
    return classes, [wnids[i] for i in classes]


def load_model(args):
    if args.ckpt:
        return load_fp32(args.data, args.ckpt)
    a = json.load(open(os.path.join(args.qat_run, "args.json")))
    wbits = a["wbits"] if a["wbits"] is not None else \
        {k: int(v) for k, v in json.load(open(a["bitmap"]))["bitmap"].items()}
    m = quantize_model(load_fp32(args.data, a["ckpt"], fold=True), wbits, a["abits"])
    m.load_state_dict(torch.load(os.path.join(args.qat_run, "best.pt"), map_location="cpu"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["patterns", "eval"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", help="extracted ImageNet-C root")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ckpt", help="FP32 checkpoint")
    g.add_argument("--qat_run", help="train_qat.py output dir")
    ap.add_argument("--out", default="runs/imagenet_c.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "eval" and not args.root:
        ap.error("eval requires --root")
    if args.cmd == "eval" and not (args.ckpt or args.qat_run):
        ap.error("eval requires --ckpt or --qat_run")

    classes, wnids = our_wnids(args.data)
    if args.cmd == "patterns":
        p = os.path.join(args.data, "imagenet_c_patterns.txt")
        open(p, "w").write("".join(f"*/{w}/*\n" for w in wnids))
        print(f"wrote {len(wnids)} patterns to {p}"); return

    dev = device()
    model = load_model(args).to(dev).eval()
    clean = evaluate(model, get_loaders(args.data, 256, args.workers)["val"], dev)["top1"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    wnid_to_label = {w: i for i, w in enumerate(wnids)}
    tag = args.ckpt or args.qat_run
    for c in CORRUPTIONS:
        for s in range(1, 6):
            d = os.path.join(args.root, c, str(s))
            ds = datasets.ImageFolder(d, tf)
            assert sorted(ds.classes) == sorted(wnids), f"{d}: class folders != our 100 wnids"
            remap = torch.tensor([wnid_to_label[w] for w in ds.classes])
            ds.target_transform = lambda y, r=remap: int(r[y])
            r = evaluate(model, DataLoader(ds, 256, num_workers=args.workers, pin_memory=True), dev)
            row = dict(model=tag, corruption=c, severity=s, top1=r["top1"], ece=r["ece"],
                       n=r["n"], clean_top1=clean, relative=r["top1"] / clean)
            print(row, flush=True); append_jsonl(args.out, row)


if __name__ == "__main__":
    main()
