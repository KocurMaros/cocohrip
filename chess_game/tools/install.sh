#!/bin/bash
# Everything chess_game needs that is NOT in the git repository:
#   python-chess (pip, user)          Stockfish (apt)
#   lc0 = Leela Chess Zero (built from source, CPU/OpenBLAS) -> ~/chess_engines/lc0
#   nets: Maia 1100..1900 + a small Leela net             -> ~/chess_engines/
#   --venv: training venv ~/chess_venv (PyTorch + CUDA, ~3 GB download)
# Steps that are already done are skipped, so it is safe to run again.
#
#   bash chess_game/tools/install.sh            # engines + python-chess
#   bash chess_game/tools/install.sh --venv     # ... and the training venv
#   bash chess_game/tools/install.sh --dry-run  # only show what would be done
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ENGINES=~/chess_engines
VENV=~/chess_venv
LC0_TAG=v0.32.1
MAIA_URL=https://github.com/CSSLab/maia-chess/raw/master/maia_weights
LEELA_URL=https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz
DRY=0; WANT_VENV=0
for a in "$@"; do
  case $a in
    --dry-run) DRY=1 ;;
    --venv) WANT_VENV=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

run() { echo "+ $*"; if [ $DRY = 0 ]; then eval "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "== system packages (sudo)"
PKGS=""
have stockfish || [ -x /usr/games/stockfish ] || PKGS="$PKGS stockfish"
have meson || PKGS="$PKGS meson"
have ninja || PKGS="$PKGS ninja-build"
have g++ || PKGS="$PKGS g++"
dpkg -s libopenblas-dev >/dev/null 2>&1 || PKGS="$PKGS libopenblas-dev"
dpkg -s zlib1g-dev >/dev/null 2>&1 || PKGS="$PKGS zlib1g-dev"
if [ $WANT_VENV = 1 ]; then python3 -c "import ensurepip" 2>/dev/null || PKGS="$PKGS python3-venv"; fi
if [ -n "$PKGS" ]; then run sudo apt-get install -y $PKGS; else echo "   all installed"; fi

echo "== python-chess"
if python3 -c "import chess" 2>/dev/null; then echo "   installed"; else
  run python3 -m pip install --user --break-system-packages chess
fi

echo "== lc0 ($LC0_TAG)"
run mkdir -p $ENGINES
if [ -x $ENGINES/lc0 ]; then echo "   $ENGINES/lc0 exists"; else
  [ -d $ENGINES/lc0-src ] || run git clone --depth 1 --branch $LC0_TAG --recurse-submodules \
      https://github.com/LeelaChessZero/lc0.git $ENGINES/lc0-src
  run "cd $ENGINES/lc0-src && ./build.sh release -Dgtest=false"
  run cp $ENGINES/lc0-src/build/release/lc0 $ENGINES/lc0
fi

echo "== networks"
for r in 1100 1300 1500 1700 1900; do
  [ -s $ENGINES/maia-$r.pb.gz ] || run wget -q -O $ENGINES/maia-$r.pb.gz $MAIA_URL/maia-$r.pb.gz
done
[ -s $ENGINES/leela.pb.gz ] || run wget -q -O $ENGINES/leela.pb.gz $LEELA_URL
echo "   $(ls $ENGINES 2>/dev/null | grep -c pb.gz) nets in $ENGINES"

if [ $WANT_VENV = 1 ]; then
  echo "== training venv $VENV"
  if [ -x $VENV/bin/python ] && $VENV/bin/python -c "import torch" 2>/dev/null; then echo "   exists"; else
    run python3 -m venv $VENV
    run $VENV/bin/pip install -q --upgrade pip
    run $VENV/bin/pip install -q torch torchvision onnx opencv-python-headless chess pyyaml
  fi
fi

if [ $DRY = 0 ]; then
  echo "== check"
  CHESS_PKG="$HERE/.." python3 - <<'EOF'
import os, sys
import chess
sys.path.insert(0, os.environ["CHESS_PKG"])
from chess_game.engines import create_engine
for spec in ("stockfish", "maia", "leela"):
    try:
        with create_engine(spec, fallback=None) as e:
            print(f"   {spec:9s} OK  first move {e.choose_move(chess.Board(), level=3, time_limit=1)}")
    except Exception as ex:  # noqa: BLE001
        print(f"   {spec:9s} FAILED: {ex}")
EOF
fi
