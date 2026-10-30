# chess_game - chess against the UR5e

Branch `feat/chess`. On the robot PC it lives in its own workspace,
`~/chess_ws` (a clone of this repo; `~/checkers_ws` is untouched).

| Part | State |
|---|---|
| rules, move recognition, robot pick/place plans (`game.py`) | done, unit tested |
| engines: Stockfish / Leela / Maia / builtin (`engines.py`) | Stockfish installed; Leela/Maia need the `lc0` binary |
| ROS node + dashboard (`chess_node.py`) | done, simulated games pass; not run on the cell yet |
| robot_control executor (`/chess/robot_move`) | done, verified on mock hardware |
| piece classifier (`vision/classifier.py`, `tools/train_classifier.py`) | pipeline verified on the robot PC GPU; **needs the dataset** |
| dataset recorder (`tools/capture_dataset.py`) | ready, waiting for the recording session |
| piece sizes (`config/pieces.yaml`) | **estimates**, to be measured with the gripper |

## The camera

Ximea MC050CG, 2464x2056. The checkers node's 600x600 frames are too coarse,
so chess uses the raw frame, analysed at half resolution (~105 px per square).
The camera is ~86 cm up, just outside the **a8 corner**: pieces near a8 are
seen from the top, near h1 from the side, and tall pieces lean over the squares
behind them. The human sits on the printed rank-1 side, the robot on rank 8.

## How a move is recognised

1. **Upright crops** (`vision/board_model.py`): from the camera pose (solvePnP
   on the board corners) every square's vertical axis is projected. Its crop
   is rotated so that the piece stands upright with its base at the same spot.
2. **CNN** (ResNet18, ImageNet-pretrained, fine-tuned on our photos) classifies
   the 64 crops into empty + 12 pieces. It runs as ONNX in OpenCV (no torch in
   the ROS environment), only when the board has been still for `stable_time`.
3. **Likelihood matching** (`game.py`): every legal move predicts a board,
   scored over all 64 squares; the best one wins if it clearly beats the
   runner-up. Colour alone already identifies every move. A mistake is shown
   with the squares that differ and the closest legal move.
4. **Lighting**: photos in dark / normal / too bright light x 3 exposures,
   augmentation, and automatic exposure + gain on the board's white squares.

### Rules decisions

* **Colour**: the colour you set up on your side is yours (S). With black,
  the robot (white) starts. F starts from any position (C picks your colour).
* **Promotion**: there are no spare pieces. The pawn stays a pawn on the
  board and becomes a queen in the game (P cycles Q/R/B/N for your pawns); the
  dashboard marks it with an orange "P", and the robot grips it as a pawn.
* **Captures**: the robot takes your captured pieces to storage, one column of
  8 slots left of the board, then one right of it (as in checkers). You take
  the robot's pieces off yourself.

## Robot PC

```bash
# build (once / after git pull) - only ROS Jazzy sourced
cd ~/chess_ws && colcon build
# run: robot bringup + robot_control from ~/chess_ws as for checkers, then
source ~/chess_ws/install/setup.bash
ros2 launch chess_game chess_game_launch.py engine:=stockfish level:=3
#   engine: stockfish | leela | maia (level 1..5 = Maia 1100..1900) | builtin
```

Python on the robot PC: `python-chess` (pip, user), `stockfish` (apt),
training venv `~/chess_venv` (PyTorch + CUDA). Leela / Maia need `lc0` in
`~/chess_engines/` or on PATH, plus the nets `~/chess_engines/leela.pb.gz`
and `maia-1100..1900.pb.gz`.

### Dataset session (with a person at the board)

```bash
python3 ~/chess_ws/src/cocohrip/chess_game/tools/capture_dataset.py --lighting normal
```

The window says what to do ("Move the BLACK knight g8 -> f6"), shows the
target position, and highlights the squares (red = take from, green = put on).
Do it and take your hand away. When the board is still and changed exactly as
asked, 3 photos are taken automatically. Resets (empty board, start position)
wait for SPACE. Change the light and press `L` to switch the tag (normal ->
dark -> bright). Aim for ~300 positions per lighting (~20-25 min each). `U`
deletes and retakes wrong photos, and `--resume <dir>` continues a session.

### Training and checking

```bash
~/chess_venv/bin/python ~/chess_ws/src/cocohrip/chess_game/tools/train_classifier.py \
    "~/chess_dataset/*" --out ~/chess_models/latest          # ~ minutes on the RTX 2060
python3 ~/chess_ws/src/cocohrip/chess_game/tools/eval_classifier.py \
    ~/chess_models/latest/piece_classifier.onnx ~/chess_dataset/<session> --show
```

`eval_classifier` reports accuracy by lighting/exposure and whether every
recorded game move is recognised from the classifier's probabilities.

### Robot side without the robot

`bash chess_game/tools/mock_robot/run.sh` starts the UR driver on mock
hardware, MoveIt, a fake gripper and robot_control in an isolated ROS domain.
It sends a capture plus castling and waits for `robot_move_done`.

## Tests (no ROS, no camera)

```bash
cd chess_game && python3 -m pytest test -q    # needs: pip install chess pytest
```
`test_node_sim.py` plays random games through the node against a fake robot
and a fake camera, in both colours.
