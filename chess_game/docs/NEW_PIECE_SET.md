# Using a different chess set

Two things depend on the pieces:

1. their **sizes**: where the gripper holds them and how wide they are there
   (`config/pieces.yaml`);
2. their **look**: the piece classifier
   (`~/chess_models/<name>/piece_classifier.onnx`).

The board, camera, rules, engines and robot calibration stay the same. A
different *board* is a separate job: re-measure `board_h8_xy` / `board_a1_xy`
for robot_control, and if its squares are not 56.6 mm, change `SQUARE_M` in
`vision/board_model.py`.

## Does it work without retraining?

Partly. The game only needs to know, per square, **empty / white / black**:
every legal move changes that pattern differently. Telling a light piece from a
dark one usually carries over to a new set if it is also clearly light vs dark.
Piece types (bishop vs pawn) usually do not carry over.

* **Colour-only mode**: `colour_only:=true` ignores the piece types. You can
  play normally (S start, moves, captures, castling, en passant). Not available:
  F (start from any position) and the "wrong piece on b1" check. Expect lower
  confidence: the camera will more often say "not sure", and you fix it by
  centring the piece or pressing SPACE.
* **Measure it instead of guessing** (step 2 below): 30 photos with the new set
  and `eval_classifier` show square accuracy and the share of moves recognised.

## Steps

### 1. Sizes with the gripper (robot on, ~10 min)

Requirements: at the grip point the pieces must be narrower than ~40 mm, and
the bases narrower than the Hand-E's 50 mm opening. A piece taller than ~36 mm
plus its grip height would touch the gripper body (46.5 mm above the
fingertips); the tool keeps 10 mm clearance automatically.

1. Clear the board. Put one **white** K, Q, R, B, N, P centred on
   c3, e3, g3, c5, e5, g5. Stand at the E-stop.
2. Robot bringup and robot_control running as usual (the tool reads the
   board corners from robot_control and moves the arm through MoveIt at
   5-10 % speed).
3. Run:
   ```bash
   cd ~/chess_ws/src/cocohrip/chess_game/tools
   source ~/chess_ws/install/setup.bash
   python3 measure_pieces.py check                # no motion: services, targets
   python3 measure_pieces.py hover c3             # 15 cm above the king: is it centred?
   python3 measure_pieces.py measure              # ~5 min, scan of all six
   mkdir -p ~/chess_sets/<name>
   python3 measure_pieces.py yaml ~/chess_sets/<name>/pieces.yaml
   ```
4. Copy the grip heights from that yaml into `LIFT_GRIPS` in
   `measure_pieces.py`, then run `python3 measure_pieces.py lift`. Every piece
   must report "lifted 3 cm and put back".

The knight is gripped round its body near the base. A knight whose head
sticks out sideways at that height would need the gripper turned; that is not
supported yet.

### 2. Does the classifier know the new set? (no robot, ~10 min)

```bash
python3 ~/chess_ws/src/cocohrip/chess_game/tools/capture_dataset.py --lighting normal
#   ~30 positions, then Q
python3 ~/chess_ws/src/cocohrip/chess_game/tools/eval_classifier.py \
    ~/chess_models/latest/piece_classifier.onnx ~/chess_dataset/<that session>
```

| "moves recognised" | what to do |
|---|---|
| >= 99 % | done, use the model as it is |
| colours right, piece types wrong | play with `colour_only:=true`, or fine-tune (step 3) |
| lower | fine-tune (step 3) |

### 3. Fine-tune (~30 min recording + a few minutes training)

Record ~100 positions per lighting (normal, dark, bright) with
`capture_dataset.py`. Then fine-tune the existing model:

```bash
~/chess_venv/bin/python ~/chess_ws/src/cocohrip/chess_game/tools/train_classifier.py \
    "~/chess_dataset/<new set sessions>*" \
    --init ~/chess_models/latest/piece_classifier.pt --lr 1e-4 --epochs 10 \
    --out ~/chess_models/<name>
```

To get one model for both sets, pass the old and the new sessions together
(without `--init`, or with it). Check with `eval_classifier.py` again.

### 4. Play

```bash
ros2 launch chess_game chess_game_launch.py \
    model_path:=~/chess_models/<name>/piece_classifier.onnx \
    pieces_yaml:=~/chess_sets/<name>/pieces.yaml          # [colour_only:=true]
```
