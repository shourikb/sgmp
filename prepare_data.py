import argparse, json, os

from datasets import load_dataset
from PIL import Image
from timm.data import ImageNetInfo


def resize_short(img, short=256):
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return img


def in1k_descriptions():
    info = ImageNetInfo()
    try:
        return list(info.label_descriptions())
    except AttributeError:
        return [info.index_to_description(i) for i in range(1000)]


def build_label_map(names):
    descs = in1k_descriptions()
    full = {d.strip().lower(): i for i, d in enumerate(descs)}
    first = {}
    for i, d in enumerate(descs):
        first.setdefault(d.split(",")[0].strip().lower(), []).append(i)
    out = {}
    for lbl, n in enumerate(names):
        key = n.strip().lower()
        head = key.split(",")[0].strip()
        if key in full:
            out[lbl] = full[key]
        elif len(first.get(head, [])) == 1:
            out[lbl] = first[head][0]
        else:
            raise ValueError(f"could not map {n!r} to an ImageNet-1k index; add it by hand")
    assert len(set(out.values())) == 100, "two labels mapped to the same 1k index"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_name", default="clane9/imagenet-100")
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    ds = load_dataset(args.hf_name)
    names = ds["train"].features["label"].names
    assert len(names) == 100, f"expected 100 classes, got {len(names)}"
    label_to_idx = build_label_map(names)

    for split, sub in [("train", "train"), ("validation", "val")]:
        split_dir = os.path.join(args.out, sub)

        def save(ex, idx):
            d = os.path.join(split_dir, f"{label_to_idx[ex['label']]:04d}")
            os.makedirs(d, exist_ok=True)
            resize_short(ex["image"]).save(os.path.join(d, f"{idx}.jpg"), quality=95)
            return {}

        print(f"writing {split} ({len(ds[split])} images) -> {split_dir}", flush=True)
        ds[split].map(save, with_indices=True, num_proc=args.num_proc,
                      remove_columns=ds[split].column_names, desc=split)

    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(sorted(label_to_idx.values()), f)
    print("done")


if __name__ == "__main__":
    main()
