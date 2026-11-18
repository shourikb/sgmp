import json, os, random

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

def tf(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def split_indices(targets, holdout_per_class, calib_n, seed=0):
    g = np.random.default_rng(seed)
    targets = np.asarray(targets)
    holdout = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        holdout += g.choice(idx, holdout_per_class, replace=False).tolist()
    hold = set(holdout)
    train = [i for i in range(len(targets)) if i not in hold]
    calib = g.choice(train, calib_n, replace=False).tolist()
    return sorted(train), sorted(holdout), sorted(calib)


def get_loaders(root, batch_size=256, workers=8, holdout_per_class=10, calib_n=512):
    tr_aug = datasets.ImageFolder(os.path.join(root, "train"), tf(True))
    tr_eval = datasets.ImageFolder(os.path.join(root, "train"), tf(False))
    val = datasets.ImageFolder(os.path.join(root, "val"), tf(False))
    tr_i, ho_i, ca_i = split_indices(tr_aug.targets, holdout_per_class, calib_n)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return {
        "train": DataLoader(Subset(tr_aug, tr_i), batch_size, shuffle=True, drop_last=True, **kw),
        "holdout": DataLoader(Subset(tr_eval, ho_i), batch_size, **kw),
        "calib": DataLoader(Subset(tr_eval, ca_i), 128, **kw),
        "val": DataLoader(val, batch_size, **kw),
    }


def build_resnet18_100(classes_json, pretrained=True, arch="resnet18.tv_in1k"):
    m = timm.create_model(arch, pretrained=pretrained)
    with open(classes_json) as f:
        classes = json.load(f)
    old = m.get_classifier()
    new = torch.nn.Linear(old.in_features, len(classes))
    with torch.no_grad():
        new.weight.copy_(old.weight[classes]); new.bias.copy_(old.bias[classes])
    m.fc = new
    return m


def load_fp32(data_root, ckpt, fold=False):
    from quant import fold_bn
    m = build_resnet18_100(os.path.join(data_root, "classes.json"), pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return fold_bn(m) if fold else m

def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return ece.item()


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None, return_logits=False):
    model.eval()
    logits, labels = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits.append(model(x.to(dev, non_blocking=True)).float().cpu()); labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    probs = logits.softmax(1)
    out = {
        "top1": probs.argmax(1).eq(labels).float().mean().item() * 100,
        "nll": F.cross_entropy(logits, labels).item(),
        "ece": ece_score(probs, labels) * 100,
        "n": len(labels),
    }
    if return_logits:
        out["_logits"], out["_labels"] = logits, labels
    return out


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
