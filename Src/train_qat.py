import argparse, json, math, os, time

import torch
import torch.nn.functional as F

from common import append_jsonl, device, evaluate, get_loaders, load_fp32, seed_all
from quant import calibrate_activations, init_weight_scales, quantize_model, size_mb


def kd_loss(s_s, s_t, y, alpha, tau):
    ce = F.cross_entropy(s_s, y)
    if alpha == 0:
        return ce
    kl = F.kl_div(F.log_softmax(s_s / tau, 1), F.log_softmax(s_t / tau, 1),
                  log_target=True, reduction="batchmean")
    return (1 - alpha) * ce + alpha * tau ** 2 * kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wbits", type=int)
    g.add_argument("--bitmap", help="json from sgmp.py")
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None, help="default 0.01, or 0.001 at 8 bits")
    ap.add_argument("--wd", type=float, default=2.5e-5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=2, help="checkpoint (and val-eval) every k epochs")
    ap.add_argument("--select_on", default="holdout", choices=["holdout"],
                    help="best snapshot is chosen on holdout; val is test-only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop early")
    ap.add_argument("--log_every", type=int, default=50, help="print per-step diagnostics every k steps")
    ap.add_argument("--clip_grad", type=float, default=0.0, help="max grad norm for weights (0 = off)")
    ap.add_argument("--lr_s_mult", type=float, default=1.0, help="step-size lr = lr * this")
    args = ap.parse_args()

    seed_all(args.seed); dev = device(); os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    log = os.path.join(args.out, "log.jsonl")
    L = get_loaders(args.data, args.bs, args.workers)

    teacher = load_fp32(args.data, args.ckpt).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    wbits = args.wbits if args.wbits is not None else \
        {k: int(v) for k, v in json.load(open(args.bitmap))["bitmap"].items()}
    lr = args.lr if args.lr is not None else (0.001 if args.wbits == 8 else 0.01)
    student = quantize_model(load_fp32(args.data, args.ckpt, fold=True), wbits, args.abits).to(dev)
    init_weight_scales(student, "mse")
    calibrate_activations(student, L["calib"], dev, "mse")
    sz = size_mb(student)

    r = evaluate(student, L["holdout"], dev)
    r.update(epoch=-1, split="holdout", size_mb=sz); print("init (=PTQ-mse):", r, flush=True); append_jsonl(log, r)
    best, best_epoch = r["top1"], -1
    torch.save(student.state_dict(), os.path.join(args.out, "best.pt"))

    w_s = [p for n, p in student.named_parameters() if n.endswith("wq.s")]
    a_s = [p for n, p in student.named_parameters() if n.endswith("aq.s")]
    step_sizes = w_s + a_s
    weights = [p for n, p in student.named_parameters() if not n.endswith(".s")]
    opt = torch.optim.SGD([{"params": weights, "weight_decay": args.wd, "lr": lr},
                           {"params": step_sizes, "weight_decay": 0.0, "lr": lr * args.lr_s_mult}],
                          lr=lr, momentum=0.9)
    total = args.epochs * len(L["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / total)))

    step = 0
    for ep in range(args.epochs):
        student.train(); t0 = time.time(); run, nb = 0.0, 0
        for x, y in L["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.no_grad():
                s_t = teacher(x) if args.alpha > 0 else None
            s_s = student(x)
            loss = kd_loss(s_s, s_t, y, args.alpha, args.tau)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}; lower --lr or set --clip_grad")
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(weights, args.clip_grad if args.clip_grad > 0 else float("inf"))
            opt.step(); sched.step()
            for p in step_sizes:
                p.data.clamp_(min=1e-8)
            run += loss.item(); nb += 1; step += 1
            if args.log_every and (step <= 5 or step % args.log_every == 0):
                with torch.no_grad():
                    ws, as_ = torch.stack([p.detach() for p in w_s]), torch.stack([p.detach() for p in a_s])
                    acc = s_s.argmax(1).eq(y).float().mean().item() * 100
                print(f"step {step:5d} loss {loss.item():.4f} batch_acc {acc:5.1f} logit_std {s_s.std().item():.3f} "
                      f"gnorm {gnorm.item():.3g} lr {sched.get_last_lr()[0]:.2e} "
                      f"w_s[{ws.min().item():.2e},{ws.max().item():.2e}] a_s[{as_.min().item():.2e},{as_.max().item():.2e}]",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        h = evaluate(student, L["holdout"], dev)
        row = dict(epoch=ep, split="holdout", train_loss=run / max(1, nb),
                   sec=time.time() - t0, **h)
        print(row, flush=True); append_jsonl(log, row)
        if h["top1"] > best:
            best, best_epoch = h["top1"], ep
            torch.save(student.state_dict(), os.path.join(args.out, "best.pt"))
        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            torch.save(student.state_dict(), os.path.join(args.out, f"ep{ep + 1}.pt"))
            v = evaluate(student, L["val"], dev)
            append_jsonl(log, dict(epoch=ep, split="val_checkpoint", **v))
        if args.max_steps and step >= args.max_steps:
            break

    student.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    r = evaluate(student, L["val"], dev, return_logits=True)
    torch.save({"logits": r.pop("_logits"), "labels": r.pop("_labels")}, os.path.join(args.out, "val_logits.pt"))
    if best_epoch == -1:
        print("WARNING: QAT never beat its PTQ initialisation on holdout; best.pt = PTQ init", flush=True)
    r.update(split="val_final", size_mb=sz, best_epoch=best_epoch, lr=lr,
             **{k: vars(args)[k] for k in ("wbits", "bitmap", "abits", "alpha", "tau", "seed")})
    print(r); append_jsonl(log, r)
    json.dump(r, open(os.path.join(args.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
