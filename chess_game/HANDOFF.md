# Chess: handoff (updated 2026-11-03)

Goal: the checkers app, but chess. The human plays against the UR5e with a
Hand-E gripper; a Ximea camera reads the board. Branch **`feat/chess`**
(based on main, which includes adaptive_pieces). Design and commands:
[README.md](README.md). Start / play / stop on the cell: [docs/STARTUP.md](docs/STARTUP.md).
Another piece set: [docs/NEW_PIECE_SET.md](docs/NEW_PIECE_SET.md).

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

* **Dataset** (2026-11-03, cell camera): `~/chess_dataset/20261103_130455` on
  the robot PC, a copy on the laptop that ran the session
  (`~/chess_dataset_backup/`). 459 positions x 3 exposures = 1,377 photos,
  1.0 GB: dark 169, normal 141, bright 149 positions (the bright ones are
  under artificial light). Labels were cleaned with `tools/check_dataset.py`
  (rebuilt runs of photos taken before the move, model-based fixes).
  `labels.before_*.jsonl` are the backups of each step.
* **Piece classifier**: `~/chess_models/latest` -> `final_v1` (ResNet18, basic
  augmentation, 45 MB, not in git, also backed up on the session laptop). On
  the held-out photos every square is right.
  * `final_v2` (cross-checked labels, strong augmentation): every square's
    colour right and 100 % of held-out moves recognised, but in light it has
    never seen the strong augmentation loses colour accuracy.
  * `final_v3` (cross-checked labels + basic augmentation) was training at the
    end of 2026-11-03: `~/chess_models/final_v3`, log `~/chess_final_v3.log`.
    If it finished and its `--val-only` numbers are at least `final_v2`'s, point
    `latest` at it.
* **Robustness in light the model never saw** (leave-one-lighting-out,
  `tools/lighting_test.sh`): squares 98.7-99.7 %, but almost all errors are
  piece types within one colour. Empty / white / black stays at 99.76-99.96 %.
  That is why the node matches moves on colours (`match_on_colours`, default)
  and reads every board from 3 frames, each also mirrored. Matching on colours
  recognised 100 % of the moves in unseen artificial light (50 % with piece
  types).

Not done yet: the chess node has never run on the cell, and no chess move
has been executed by the real robot.

## Next steps (in order)

0. On the robot PC: `git pull`, `colcon build`, `bash chess_game/tools/install.sh`
   (now also installs onnxruntime-gpu; its last lines must say
   `classifier backend: onnxruntime-gpu`), `ln -sfn ~/chess_models/final_v3
   ~/chess_models/latest` (final_v3: cleaned labels, 100 % of held-out moves).
   Then `tools/latency_replay.py` on the PC should give about 0.6 s, and a few
   moves on the cell confirm it. The speed-ups (GPU, background reading,
   0.5 s stillness, accepting a reading whose colours changed) and the status
   spinner were only tested in replay and simulation so far.
   Stockfish levels 2..4 now use `UCI_Elo` (1350 / 1600 / 2000):
   `python3 -m pytest chess_game/test/test_engines.py` on the PC plays every
   level with the real Stockfish (not installed on the dev laptop).
0b. **Desktop icons** (branch `feat/launcher`, contains all of feat/chess):
   `bash launcher/install.sh` in the desktop session, then try **Chess robot**
   with the robot. Check that the robot rows (power, brakes, safety, pendant
   program) follow the pendant, that the parts start in order and that **Stop**
   stops them. It was tested only with fake ROS nodes so far.
   Then **Checkers robot** once. See [../launcher/README.md](../launcher/README.md).
1. **First game on the cell** (see [docs/STARTUP.md](docs/STARTUP.md)): `ros2 launch chess_game chess_game_launch.py` (it
   loads `~/chess_models/latest`). Check move recognition live, with hands and
   lighting changes.
2. **Robot on the cell**: start bringup and `robot_control` from `~/chess_ws`
   (only that build has `/chess/robot_move`). Send single moves first and check
   the squares, storage slots (one column left, one right of the board), placing
   and travel height.
3. **Full games**: captures, castling, en passant, promotion, wrong moves,
   hands over the board, lighting changes, failed grips.
4. Piece centring from the camera (`PickPlace.offset_*` is always 0 now).
5. Gripper rotation by the direction the knight's head faces (asked for by the user).
5b. **Learn from real games**: save the photos of positions the game confirmed
   (the labels come free from the game state) and retrain now and then. The
   model then adapts to new light without a recording session.
6. **Player colour**: it must follow from where the pieces stand. That is
   implemented: S detects the colour on the human's side (black = the robot
   starts), and C picks it for F. Check it on the cell with both set-ups.
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
  `/opt/XIMEA/bin/xiSample` has run once. `ChessCamera` retries. It must sit
  on a **USB 3** port: at USB 2 (`lsusb -t` shows 480M) it delivers no
  frames.
* **Dataset recording lessons**: a capture done the wrong way round, or a
  wrong piece from the box, is invisible to the change detection. The live
  check with a trained model (`--check-model`) catches most of them, but its
  king/queen opinions in new light are unreliable. SPACE pressed during the
  photos used to label the board as the next position (fixed).
* **Retraining**: `bash tools/train_pipeline.sh <session> <name>` (clean,
  train, held-out evaluation); for a new piece set see
  [docs/NEW_PIECE_SET.md](docs/NEW_PIECE_SET.md).
* **Tracked pycache**: `checkers_game/.../camera/__pycache__/*.pyc` is tracked
  in git and changes whenever tests import checkers_game. Don't commit it.
* **Plain-text token**: the robot PC's `~/checkers_ws/src` remote `upstream` has
  a GitHub token in its URL.
* **Mock robot test**: keep it in its own ROS domain with a fake robot IP (the
  script does this) so it can never talk to the real robot.
