#!/bin/bash
# Cross-checked label cleaning: a model trained on the even blocks of 25
# positions checks (and fixes) the odd ones, and the other way round - a
# model never judges photos it was trained on, so it cannot have learned
# their mistakes.  Then trains the final model on everything.
#   setsid nohup bash chess_game/tools/crossval_clean.sh ~/chess_dataset/<session> <name> > ~/chess_crossval.log 2>&1 &
set -euo pipefail
S=${1:?session directory}
NAME=${2:-final_$(date +%Y%m%d_%H%M)}
T=$(cd "$(dirname "$0")" && pwd)
PY=~/chess_venv/bin/python
cp "$S/labels.jsonl" "$S/labels.before_crossval_$(date +%H%M%S).jsonl"
for half in even odd; do
  other=$([ $half = even ] && echo odd || echo even)
  echo "== $(date +%T) model on the $half blocks"
  $PY "$T/train_classifier.py" "$S" --out ~/chess_models/${NAME}_cv_$half --train-blocks $half \
      --epochs 12 --workers 6 2>&1 | grep -E "^epoch|photos" | tail -2
  echo "== $(date +%T) it checks the $other blocks"
  python3 "$T/check_dataset.py" "$S" --model ~/chess_models/${NAME}_cv_$half/piece_classifier.onnx \
      --blocks $other --fix | tail -60
done
echo "== $(date +%T) final model on all photos"
$PY "$T/train_classifier.py" "$S" --out ~/chess_models/$NAME --epochs 15 --workers 6 2>&1 \
    | grep -v -i warn | tail -40
ln -sfn ~/chess_models/$NAME ~/chess_models/latest
echo "== $(date +%T) held-out evaluation"
python3 "$T/eval_classifier.py" ~/chess_models/$NAME/piece_classifier.onnx "$S" --val-only
echo "== $(date +%T) CROSSVAL DONE -> ~/chess_models/$NAME"
