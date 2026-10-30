# Chess: handoff (2026-10-30)

Goal: the checkers app, but chess. The human plays against the UR5e with a
Hand-E gripper; a Ximea camera reads the board. Branch **`feat/chess`**
(based on main, which includes adaptive_pieces). Design and commands:
[README.md](README.md). Another piece set: [docs/NEW_PIECE_SET.md](docs/NEW_PIECE_SET.md).

## Where things are

| | |
|---|---|
| code | `chess_game/` (Python, node + tools), `chess_msgs/`, `robot_control/` (new `/chess/robot_move` subscriber) |
| robot PC | `ssh collab@147.175.115.82` (password from the team) |
| chess workspace | `~/chess_ws` = clone of this repo on `feat/chess`, fully built. `~/checkers_ws` (the team's checkers setup) is untouched |
| not in git | `~/chess_engines` (lc0 + Maia/Leela nets), Stockfish (apt), python-chess (pip --user), `~/chess_venv` (PyTorch, training). All of it: `bash chess_game/tools/install.sh [--venv]` |
| data (later) | `~/chess_dataset/<session>/`, `~/chess_models/<name>/`, board corners cache `~/.cocohrip/chess_board.json` |

## State

Done and verified:
* **Rules and move recognition** (`game.py`, python-chess): unit tests.
  `test_node_sim.py` plays random games through the node against a fake robot.
* **Engines**: Stockfish, Leela and Maia (lc0 on CPU) and any `uci:/path`
  engine all answer on the robot PC.
* **robot_control chess executor**: on mock hardware (`tools/mock_robot/run.sh`)
  a capture to storage plus castling ran through and reported done. Found and
  fixed a pre-existing bug on the way: `robot_move_done` was never sent after
  the final HOME (commit `9c2e372`, affects checkers too).
* **Piece sizes**: measured with the gripper and lift-tested on the cell
  (`config/pieces.yaml`, `tools/measure_pieces.py`).
* **Camera and classifier pipeline**: camera, board detection and pose checked
  live. Training, ONNX export and OpenCV inference run on the robot PC (RTX
  2060) with fake data.

Not done yet: the classifier has **no real training data**, so the node
cannot play yet. The chess node has never run on the cell, and no chess move
has been executed by the real robot.

## Next steps (in order)

1. **Dataset** (needs a person at the board, ~1.5 h): `tools/capture_dataset.py`
   in three lightings (`--lighting normal`, then L for dark and bright), about 300
   positions each. Then `train_classifier.py` in `~/chess_venv` and
   `eval_classifier.py`. Target: >= 99.5 % of moves recognised. Commands are in
   the README.
2. **Robot on the cell**: start bringup and `robot_control` from `~/chess_ws`
   (only that build has `/chess/robot_move`). Send single moves first and check
   the squares, storage slots (one column left, one right of the board), placing
   and travel height.
3. **Full games**: captures, castling, en passant, promotion, wrong moves,
   hands over the board, lighting changes, failed grips.
4. Piece centring from the camera (`PickPlace.offset_*` is always 0 now).
5. Gripper rotation by the direction the knight's head faces (asked for by the user).
6. **Explicit colour choice** (asked for by the user): let the player pick
   white or black, e.g. W/B keys and a `human_color: auto|white|black` launch
   parameter. The dashboard should then say how to set up the board, and S
   should insist on that colour. Today the colour is only detected from which
   pieces stand on the human's side at S (C picks it for F).
7. Re-check checkers with the `9c2e372` fix, then a PR to main.

## Decisions agreed with the user

* No spare pieces: **promotion is virtual**. The pawn stays on the board, the
  game treats it as a queen and the dashboard marks it.
* Captured human pieces go to storage **left and right of the board** (as in
  checkers). The human removes the pieces they capture.
* The human's colour is detected from the start position (their side is
  printed rank 1).
* Engines: Stockfish plus Leela/Maia. Torch (Chess.com) is not available, and
  Leela stands in for AlphaZero.
* Dataset lighting: dark room, normal, too bright.
* Board diagram draws real piece figures, not letters.

## Gotchas

* **Camera geometry**: the camera is not above the board; it sits ~86 cm up,
  outside the a8 corner. Pieces near h1 are seen from the side, near a8 from the
  top.
* **Three square conventions** (see `frames.py`): printed (camera and labels),
  logical (rules; rotated 180° when the human plays black), and robot_control's
  game frame. Printed a1 is game (7, 0).
* **Gripper body**: it is 46.5 mm above the fingertips, so the king (63 mm) and
  queen (55 mm) must be gripped high. `pieces.yaml` takes care of it; the robot's
  old fixed `z_attach` would crash into them.
* **Ximea**: after a reboot it sometimes gives no frames until
  `/opt/XIMEA/bin/xiSample` has run once. `ChessCamera` retries.
* **Tracked pycache**: `checkers_game/.../camera/__pycache__/*.pyc` is tracked
  in git and changes whenever tests import checkers_game. Don't commit it.
* **Plain-text token**: the robot PC's `~/checkers_ws/src` remote `upstream` has
  a GitHub token in its URL.
* **Mock robot test**: keep it in its own ROS domain with a fake robot IP (the
  script does this) so it can never talk to the real robot.
