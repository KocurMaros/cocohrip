#!/bin/bash
# Clean a recorded session and train the final piece classifier - runs on
# the robot PC (GPU) and survives a dropped ssh connection:
#   setsid nohup bash chess_game/tools/train_pipeline.sh ~/chess_dataset/<session> > ~/chess_pipeline.log 2>&1 &
# 1. photos whose labelled move is not visible -> excluded
# 2. model on the raw labels -> fixes labels it is sure about (excl. king/queen)
# 3. final model on the cleaned labels -> ~/chess_models/<name>, latest -> it
# 4. evaluation on the held-out photos (accuracy per lighting, moves recognised)
set -euo pipefail
S=${1:?session directory}
NAME=${2:-final_$(date +%Y%m%d_%H%M)}
T=$(cd "$(dirname "$0")" && pwd)
PY=~/chess_venv/bin/python
echo "== $(date +%T) backup of the labels"
cp "$S/labels.jsonl" "$S/labels.before_pipeline_$(date +%H%M%S).jsonl"
echo "== $(date +%T) 1/4 change check"
python3 "$T/check_dataset.py" "$S" --exclude-missing | tail -3
echo "== $(date +%T) 2/4 model on raw labels + label fixes"
$PY "$T/train_classifier.py" "$S" --out ~/chess_models/${NAME}_raw --epochs 10 --workers 6 2>&1 \
    | grep -E "^epoch|photos|temperature"
python3 "$T/check_dataset.py" "$S" --model ~/chess_models/${NAME}_raw/piece_classifier.onnx --fix | tail -40
echo "== $(date +%T) 3/4 final model"
$PY "$T/train_classifier.py" "$S" --out ~/chess_models/$NAME --epochs 15 --workers 6 2>&1 \
    | grep -v -i warn | tail -45
ln -sfn ~/chess_models/$NAME ~/chess_models/latest
echo "== $(date +%T) 4/4 evaluation on held-out photos"
python3 "$T/eval_classifier.py" ~/chess_models/$NAME/piece_classifier.onnx "$S" --val-only
echo "== $(date +%T) PIPELINE DONE -> ~/chess_models/$NAME"
