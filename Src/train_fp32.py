import argparse, json, math, os, time

import torch
import torch.nn.functional as F

from common import append_jsonl, build_resnet18_100, device, evaluate, get_loaders, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/fp32")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop early")
    ap.add_argument("--no_pretrained", action="store_true", help="smoke test only")
    args = ap.parse_args()

    seed_all(args.seed); dev = device(); os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, "log.jsonl")
    L = get_loaders(args.data, args.bs, args.workers)
    model = build_resnet18_100(os.path.join(args.data, "classes.json"), pretrained=not args.no_pretrained).to(dev)

    r = evaluate(model, L["val"], dev); r.update(stage="sliced_head_no_finetune", split="val")
    print(r); append_jsonl(log, r)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    total = args.epochs * len(L["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / total)))
    best, step = -1, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        for x, y in L["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        h = evaluate(model, L["holdout"], dev)
        row = dict(epoch=ep, split="holdout", sec=time.time() - t0, **h); print(row); append_jsonl(log, row)
        if h["top1"] > best:
            best = h["top1"]; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        if args.max_steps and step >= args.max_steps:
            break

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    r = evaluate(model, L["val"], dev, return_logits=True)
    torch.save({"logits": r.pop("_logits"), "labels": r.pop("_labels")}, os.path.join(args.out, "val_logits.pt"))
    r.update(stage="fp32_final", split="val"); print(r); append_jsonl(log, r)
    json.dump(r, open(os.path.join(args.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
