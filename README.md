# SGMP: Sensitivity-Guided Mixed-Precision Quantization

Final project for CS 4644/7643 Deep Learning. We compress a ResNet-18 on ImageNet-100 and ask whether a
per-layer bit allocation chosen by measured loss sensitivity (SGMP) still helps after quantization-aware
training (QAT), compared with uniform precision, random allocations and a data-free heuristic.

## Setup

```bash
pip install -r requirements.txt
```

Experiments ran on a single A100 on the Georgia Tech ICE cluster (`job.sbatch` wraps any command in a Slurm job;
set `DATA` first). Jobs must be pinned to A100/H100 GPUs, because our PyTorch build crashes on Blackwell nodes.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads ImageNet-100 (`clane9/imagenet-100`) and writes the folder layout plus `classes.json` |
| `common.py` | Data splits (1,000 held-out, 512 calibration), loaders, ResNet-18 head slicing, accuracy and ECE |
| `quant.py` | LSQ quantizer, straight-through estimator, BN folding, MSE and min-max scale selection |
| `train_fp32.py` | Fine-tunes the FP32 baseline (also the distillation teacher) |
| `ptq.py` | Uniform post-training quantization sweep (min-max vs MSE scales) |
| `sgmp.py` | Per-layer sensitivity, greedy and exact-DP bit allocation, random and weight-error controls |
| `train_qat.py` | LSQ QAT with cross-entropy plus distillation (Eq. 3 in the report) |
| `imagenet_c.py` | ImageNet-C robustness evaluation, restricted to our 100 classes |
| `latency.py` | INT8 vs FP32 CPU latency |
| `analyze.py` | Builds the figures and tables from `runs/` |
| `check.sh`, `diag_qat.sh` | Cluster job status summary and the QAT learning-rate diagnostic |
| `smoke_test.py` | Quick end-to-end check on tiny data |

## Run order

```bash
export DATA=/path/to/imagenet100

python prepare_data.py --out $DATA
python train_fp32.py   --data $DATA                                  # -> runs/fp32/best.pt
python ptq.py          --data $DATA --ckpt runs/fp32/best.pt         # -> runs/ptq/ptq.jsonl
python sgmp.py         --data $DATA --ckpt runs/fp32/best.pt         # -> runs/sgmp/ (bitmaps, allocations)

# QAT, uniform vs SGMP at the 4-bit size
python train_qat.py --data $DATA --ckpt runs/fp32/best.pt --wbits 4 --lr 1e-4 --out runs/qat/u4_s0
python train_qat.py --data $DATA --ckpt runs/fp32/best.pt --bitmap runs/sgmp/bitmap_sgmp_greedy_u4.json --lr 1e-4 --out runs/qat/sgmp4_s0

python imagenet_c.py eval --data $DATA --root /path/to/ImageNet-C --qat_run runs/qat/sgmp4_s0
python latency.py      --data $DATA --ckpt runs/fp32/best.pt
python analyze.py      --runs runs --out figs
```

## Important: QAT learning rate

`train_qat.py` defaults to `--lr 0.01`, which collapses to chance accuracy on the BN-folded network.
Always pass `--lr 1e-4`, the value used for every reported result. `diag_qat.sh` reproduces the 300-step
diagnostic that picked it (lr 1e-2 and 1e-3 diverged; 1e-4 trained stably).

## Defaults used in the report

- Distillation: alpha = 0.7, temperature tau = 4
- SGD, momentum 0.9, cosine decay, 10 epochs, 8-bit activations, 8-bit first and last layers
- Test set: the 5,000 official ImageNet validation images of our 100 classes

`runs/`, `figs/` and `logs/` are git-ignored, so results are not stored in the repo.
