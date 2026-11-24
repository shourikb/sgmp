import json, os, subprocess, sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
D = "smoke_data"


def make_data():
    rng = np.random.default_rng(0)
    classes = sorted(rng.choice(1000, 100, replace=False).tolist())
    for split, n in [("train", 16), ("val", 2)]:
        for c in classes:
            d = os.path.join(D, split, f"{c:04d}"); os.makedirs(d, exist_ok=True)
            for i in range(n):
                Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(f"{d}/{i}.jpg")
    with open(os.path.join(D, "classes.json"), "w") as f:
        json.dump(classes, f)


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(D, "classes.json")):
        make_data()
    common = ["--data", D, "--workers", "0"]
    run("train_fp32.py", *common, "--out", "smoke_runs/fp32", "--epochs", "1",
        "--max_steps", "2", "--bs", "64", "--no_pretrained")
    ck = ["--ckpt", "smoke_runs/fp32/best.pt"]
    run("ptq.py", *common, *ck, "--out", "smoke_runs/ptq")
    run("sgmp.py", *common, *ck, "--out", "smoke_runs/sgmp", "--n_random", "2", "--budgets", "4,3.5")
    run("train_qat.py", *common, *ck, "--out", "smoke_runs/qat_u4", "--wbits", "4",
        "--epochs", "1", "--max_steps", "2", "--bs", "64")
    run("train_qat.py", *common, *ck, "--out", "smoke_runs/qat_sgmp4",
        "--bitmap", "smoke_runs/sgmp/bitmap_sgmp_greedy_u4.json", "--epochs", "1", "--max_steps", "2", "--bs", "64")
    run("analyze.py", "--runs", "smoke_runs", "--out", "smoke_runs/figs",
        "--qat_uniform", "qat_u4", "--qat_sgmp", "qat_sgmp4")
    print("\nSMOKE TEST PASSED")
