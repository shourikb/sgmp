#!/bin/bash
set -u
D=${DATA:?set DATA to the ImageNet-100 root written by prepare_data.py}
CKPT=${CKPT:-runs/fp32/best.pt}
common="--data $D --ckpt $CKPT --wbits 4 --epochs 1 --max_steps 300 --log_every 25"
python train_qat.py $common --lr 1e-2                 --out runs/diag/lr1e-2
python train_qat.py $common --lr 1e-2 --lr_s_mult 0   --out runs/diag/lr1e-2_frozen_s
python train_qat.py $common --lr 1e-3                 --out runs/diag/lr1e-3
python train_qat.py $common --lr 1e-4                 --out runs/diag/lr1e-4
grep -h '"split": "holdout"' runs/diag/*/log.jsonl
