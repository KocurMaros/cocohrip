# chess_game - chess against the UR5e

Status: **foundation only** (branch `feat/chess`). Done: rules glue, move
inference from the camera, robot pick/place planning, swappable engines, 3-D
board model, guided dataset recorder, `chess_msgs` draft. Next: piece
classifier (needs the dataset), the ROS node + dashboard, and the robot_control
executor.

## What the camera sees (measured 2026-10-30 on the cell)

* Ximea MC050CG, 2464x2056. The checkers node squeezes frames to 600x600
  (~50 px per square). That's too coarse for chess, so the chess code works on the raw
  frame (half resolution = ~105 px per square).
* The camera is **not above the board**. The pose fitted to the board corners puts it
  ~86 cm up, just outside the **a8 corner** (`BoardModel.camera_position`).
  Pieces near a8 are seen from the top. Towards h1 they are seen more and more from
  the side, and tall pieces lean over the squares behind them (away from a8).
* The checkers corner detector (`checkers_game.camera.board_geometry`) finds
  the board even with all 32 pieces on it (47/49 junctions). Printed a1 is the
  raw image's bottom-right corner.
* The human sits on the printed rank-1 side, the robot on rank 8 (same as checkers).

## Design

```
Ximea -> BoardModel (pose, square crops) -> classifier (ONNX) -> per-square probabilities
      -> ChessGame.infer_human_move  (likelihood of every legal move)
      -> Engine (Stockfish / lc0+Maia / builtin)  -> ChessGame.plan_robot_move
      -> chess_msgs/ChessRobotMove (pick/place steps) -> robot_control -> RobotMove done
```

### Detection

1. **Upright square crops** (`vision/board_model.py`). From the camera pose, each
   square's vertical axis is projected into the image. The crop around it is rotated
   so that the piece always stands upright with its base at the same spot, so a
   bishop looks the same on a8 and on h1.
2. **One CNN classifies all 64 crops** into 13 classes (empty + 12 pieces). Plan:
   a small ImageNet-pretrained net (MobileNetV3 / ResNet18) fine-tuned on our own
   photos, exported to ONNX. It runs with OpenCV's DNN module, which is already on
   the robot PC, so no torch is needed there. It only runs when the board is still.
3. **No single square is trusted** (`game.py`). Every legal move predicts a board,
   and each prediction is scored by the log-likelihood of all 64 squares; the best
   one wins if it beats the runner-up clearly. Colour alone (empty / white / black)
   already identifies every legal move, castling and en passant included. Piece
   types are only needed for the promotion piece, starting from an arbitrary
   position, and checking the set-up ("knight and bishop swapped on b1/c1").
4. **Lighting**: photos under several lightings x 3 exposures (bracketing),
   strong colour / gamma / white-balance / noise augmentation, plus the existing
   software auto-exposure on the board's white squares.

The labels need no manual annotation: every photo is stored with the FEN of the
physical board and the board corners, so the 64 crops and their classes follow
from that. The same data could also train a YOLO detector (boxes from the
projected piece cylinders) if the per-square classifier turns out not to be enough.

### Dataset recording (`tools/capture_dataset.py`)

The tool tells you what to do: "Move the BLACK knight g8 -> f6", with the target
position drawn and the squares highlighted. You do it and take your hand away.
When the board is still and exactly those squares changed, it takes 3 photos
(3 exposures) and shows the next instruction. Positions alternate between random
legal games (white or black on your side), single pieces scattered to the squares
seen least so far (every piece on every square), and the empty board. `L` switches
the lighting tag, `U` deletes and retakes wrong photos, and `--resume` continues a
session. Target: ~300 positions per lighting (about 20-25 min), for 3-4 lightings.

### Engines (`engines.py`)

`create_engine("stockfish" | "lc0:weights=maia-1500.pb.gz" | "builtin" | "random")`,
all with `choose_move(board, level 1..5, time_limit)`. Stockfish sets "Skill
Level" from the level. lc0 with Maia weights plays like a human of a chosen
rating (a neural-network engine). `builtin` is a pure-Python alpha-beta search,
used as the fallback when no binary is installed. A PyTorch policy network would
be one more `Engine` subclass.

### Robot (`game.plan_robot_move`, `chess_msgs`)

A robot move becomes a list of pick/place steps: the captured piece goes off first
(to a storage slot), castling moves the king then the rook, en passant removes the
passed pawn, and promotion moves the pawn off and brings the new piece from a
reserve slot (or the human swaps it). Positions are in robot_control's game frame
(`frames.py`; off-board slots are just rows/cols outside 0..7), so its corner
calibration is reused. robot_control needs a new subscriber that executes
`ChessRobotMove` steps with a **per-piece grip height and width** and a **travel
height** above the tallest piece. The current fixed `z_attach` and 9 cm travel
height would knock tall pieces over.

## Run

```bash
# tests (no ROS, no camera; needs: pip install chess pytest)
cd chess_game && python3 -m pytest test -q
# record the dataset on the robot PC (game nodes stopped, workspace sourced)
python3 chess_game/tools/capture_dataset.py --lighting room
```
