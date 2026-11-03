#!/bin/bash
# How well does the classifier handle light it has never seen?  For each
# lighting: train on the other two, evaluate on all photos of this one.
# Runs each augmentation given (default: basic and strong) and prints a table.
#   setsid nohup bash chess_game/tools/lighting_test.sh ~/chess_dataset/<session> > ~/chess_lighting.log 2>&1 &
set -uo pipefail
S=${1:?session directory}
AUGS=${2:-"basic strong"}
EPOCHS=${EPOCHS:-10}
T=$(cd "$(dirname "$0")" && pwd)
PY=~/chess_venv/bin/python
OUT=~/chess_models/lighting_test
mkdir -p $OUT
for aug in $AUGS; do
  for light in dark normal bright; do
    echo "== $(date +%T) train without $light, augmentation $aug"
    $PY "$T/train_classifier.py" "$S" --out $OUT/${aug}_no_$light --aug $aug --holdout-lighting $light \
        --epochs $EPOCHS --workers 6 2>&1 | grep -E "^epoch|photos" | tail -2
    echo "RESULT $aug unseen=$light: $(python3 "$T/eval_classifier.py" $OUT/${aug}_no_$light/piece_classifier.onnx \
        "$S" --lighting $light 2>&1 | grep -E "square accuracy|board accuracy|moves recognised" \
        | sed -E 's/  +/ /g' | tr '\n' '|')"
  done
done
echo "== $(date +%T) LIGHTING TEST DONE"
grep -h "^RESULT" ~/chess_lighting.log 2>/dev/null
