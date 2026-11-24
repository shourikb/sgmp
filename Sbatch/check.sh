#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
SINCE=${1:-$(date -d yesterday +%F)}

if command -v sacct >/dev/null; then
  echo "== jobs since $SINCE"
  sacct -u "$USER" -S "$SINCE" -X --format=JobID,JobName%10,State,ExitCode,Elapsed
fi

echo; echo "== FP32 model"
ls -la runs/fp32/best.pt 2>&1
python -c "import json; r=json.load(open('runs/fp32/final.json')); print(f'test top-1 {r[\"top1\"]:.2f}%  ECE {r[\"ece\"]:.2f}%')" 2>&1 | tail -1

echo; echo "== PTQ sweep (default sweep writes 18 lines; BN-fold diff must be below 1e-3)"
wc -l runs/ptq/ptq.jsonl 2>&1
grep -h "BN-fold" logs/*.out 2>/dev/null | tail -1

echo; echo "== SGMP (default budgets write 54 lines and 54 bitmaps; sizes at or just under budget)"
wc -l runs/sgmp/allocations.jsonl 2>&1
echo "$(ls runs/sgmp/bitmap_*.json 2>/dev/null | wc -l) bitmap files"
python -c "
import json
for r in map(json.loads, open('runs/sgmp/allocations.jsonl')):
    if not r['algo'].startswith('random'):
        print(f'  u={r[\"u\"]:<4g} {r[\"algo\"]:12s} size {r[\"size_mb\"]:.2f}/{r[\"budget_mb\"]:.2f} MB  test {r[\"val_top1\"]:.1f}%')
" 2>&1 | tail -20

echo; echo "== QAT runs: held-out top-1 per epoch (ep-1 = PTQ initialisation)"
for d in runs/qat/*/ runs/diag/*/; do
  [ -f "$d/log.jsonl" ] || continue
  printf "  %-24s " "$(basename "$d")"
  python -c "
import json, sys
R=[json.loads(l) for l in open(sys.argv[1]) if '\"holdout\"' in l]
print('  '.join(f'ep{r[\"epoch\"]}: {r[\"top1\"]:.1f}%' for r in R))
" "$d/log.jsonl" 2>&1 | tail -1
done
grep -h "non-finite" logs/*.out 2>/dev/null

echo; echo "== latency"
cat runs/latency.jsonl 2>&1

echo; echo "== logs with errors"
grep -l -E "Traceback|CANCELLED|TIMEOUT|OUT_OF_MEMORY" logs/*.out 2>/dev/null | tail -8
