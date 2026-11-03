# Chess on the cell: start, play, stop

Everything chess runs from **`~/chess_ws`** on the robot PC; checkers stays in
`~/checkers_ws`. Don't mix them in one terminal: open a new terminal for every
command, as for checkers.

The short version for the terminal is `runing_seq_chess.txt` in the repository
root (like `runing_seq.txt` for checkers).

## 0. Once (or after `git pull`)

```bash
cd ~/chess_ws/src/cocohrip && git pull
cd ~/chess_ws && colcon build                      # only ROS Jazzy sourced
bash ~/chess_ws/src/cocohrip/chess_game/tools/install.sh   # engines, GPU runtime, skips what is there
ln -sfn ~/chess_models/final_v3 ~/chess_models/latest      # the piece classifier the game loads
```

`install.sh` ends by printing the classifier backend: it should say
`onnxruntime-gpu`. With `onnxruntime-cpu` or `opencv` the game still works,
but it reads each board in ~0.5 s instead of ~0.05 s.

## 1. Robot and camera (person at the cell)

1. Power on the UR5e, release the brakes, start the External Control program
   on the pendant (as for checkers). The Hand-E gets power with the robot.
2. **Camera on USB 3:** `lsusb -t | grep -A2 -i "Miscellaneous"` must show
   `5000M`. If it shows `480M`, use another blue USB 3 port.
3. Only after a reboot, if the camera gives no frames:
   `sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0`,
   then run `/opt/XIMEA/bin/xiSample` once.

## 2. Start the software

**Easiest:** double-click **Chess robot** on the desktop. The panel starts
the three parts below in order, says what the robot still needs (brakes, ▶
on the pendant, ...) and lets you choose the opponent and level; **Stop**
ends it all. See `launcher/README.md`; to install the icons once,
run `bash ~/chess_ws/src/cocohrip/launcher/install.sh` in the desktop session.

By hand, in 3 terminals:

```bash
# terminal 1 - robot driver, MoveIt, RViz, gripper
source ~/chess_ws/install/setup.bash && ros2 launch cocohrip_control bringup_all.launch.py
# terminal 2 - robot_control (this build also executes chess moves, /chess/robot_move)
source ~/chess_ws/install/setup.bash && ros2 launch robot_control new_launch.py
# terminal 3 - the chess game (camera, piece recognition, engine, window "Chess")
source ~/chess_ws/install/setup.bash && ros2 launch chess_game chess_game_launch.py engine:=stockfish level:=3
```

Engine choices (the window shows the level's rough Elo):
- `engine:=builtin`: simple search, ~700..1300.
- `engine:=maia`: plays like a human; levels 1..5 = rating 1100..1900.
- `engine:=stockfish`: ~1000, 1350, 1600, 2000, full strength.
- `engine:=leela`: strong neural network, ~2000..3100.

In the game, **E** switches the engine and **1..5** the level. The node loads the
piece classifier from `~/chess_models/latest`, and finds the board by itself
(the corners are cached in `~/.cocohrip/chess_board.json`). If it cannot, it
asks you to click the corners of a1, a8, h8, h1.

## 3. Play

1. Set up the pieces. Your colour on your side (the printed rank-1 side,
   opposite the robot) is your colour. With black on your side the robot (white)
   starts.
2. Press **S** in the Chess window. If a piece type looks doubtful (e.g. knight
   and bishop swapped), the game starts anyway and names the squares to check.
3. Your move: move the piece, take your hand away. About half a second
   later the move appears on the game board. At the right of the banner,
   "reading the board", "robot is thinking" and "robot is moving" (with a
   turning circle) show what the system is doing. If you capture, take the robot's piece off
   yourself.
4. The robot moves. Keep your hands off the board until it is home again. It
   takes your captured pieces to the storage beside the board.
5. A pawn that reaches the last rank stays a pawn on the board, and is a queen
   in the game (orange "P" mark; **P** changes the piece for your pawns).

| Key | |
|---|---|
| S | start (start position on the board) |
| F | start from whatever position is on the board (**C** = your colour for F) |
| 1..5 | engine level (rough Elo shown) |
| E | next engine: simple, Maia, Stockfish, Leela |
| SPACE | read the board again now |
| R | the robot has finished (if it did not report) |
| L | unlocked: the game also continues when the camera sees the robot's move done |
| H | show your legal moves |
| Q | quit |

If the window says **"MOVE NOT ACCEPTED"** or **"Not sure what you played"**, it
names the squares that look wrong and the closest legal move. Put the pieces
right (centred on their squares); the board is re-read automatically.

## 4. Stop

**Q** in the Chess window, Ctrl+C in terminals 2 and 1, robot off on the pendant.

## First session with chess (with the robot)

Have someone at the E-stop for all of it.

1. **Grip and square test without the game.** With the measurement pieces
   placed (see `docs/NEW_PIECE_SET.md`), run
   `python3 tools/measure_pieces.py hover c3`. It must stop 15 cm above c3.
   Then `python3 tools/measure_pieces.py lift` grips and lifts each piece 3 cm.
2. **First robot moves in a game:** start a game where the robot moves first
   (black on your side). Watch the first moves:
   - does it grip at the right height?
   - does it put the piece in the centre of the target square?
   - does it travel high enough over the king?
3. **Captures:** the captured piece must go to the storage, a column left of
   the board, then right of it.
4. **Castling, en passant, promotion** once each.
5. Note every time the game asks you to fix something. Those are the cases to
   improve.

## Troubleshooting

| Problem | What to do |
|---|---|
| "Ximea camera delivers no frames" | USB 3 port (see 1.2), `xiSample` once, replug |
| board not found / wrong grid | the board moved: delete `~/.cocohrip/chess_board.json`, start again (auto-detect or click) |
| "no piece classifier" | `ls ~/chess_models/latest/` must contain `piece_classifier.onnx` |
| engine "cannot start ... using builtin" | `bash tools/install.sh` (Stockfish / lc0 missing) |
| robot does not react | terminal 2 must be `robot_control` from `~/chess_ws` (has `/chess/robot_move`) |
| robot finished but game waits | press **R**, or **L** for unlocked mode |
| robot could not grip a piece | it gives up the move and goes home: finish its move by hand |
| new light, many "not sure" | record ~30 positions in that light and retrain (README, "Training and checking") |

More: [README.md](../README.md) (design, training), [HANDOFF.md](../HANDOFF.md)
(state, next steps), [NEW_PIECE_SET.md](NEW_PIECE_SET.md) (another chess set).
