# Cocohrip - Checkers Robot System

A ROS2-based autonomous checkers playing robot system that uses computer vision for board detection and a UR5e robotic arm for piece manipulation.

## Table of Contents
- [System Overview](#system-overview)
- [Quick Start](#quick-start)
- [ROS2 Topics Reference](#ros2-topics-reference)
- [Message Types](#message-types)
- [Board Coordinates System](#board-coordinates-system)
- [Robot Movement Examples](#robot-movement-examples)
- [Debugging Guide](#debugging-guide)
- [Configuration](#configuration)

---

## System Overview

The system consists of three main packages:

| Package | Description |
|---------|-------------|
| `robot_control` | C++ node for UR5e robot arm control using MoveIt2 |
| `checkers_game` | Python node for game logic, AI, and computer vision |
| `checkers_msgs` | Custom ROS2 message definitions |

### Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Ximea Camera   │────▶│  checkers_game   │────▶│  robot_control  │
│  (Board Vision) │     │  (Python Node)   │     │   (C++ Node)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌──────────────┐         ┌──────────────┐
                        │  Pygame GUI  │         │  UR5e Robot  │
                        │  (Display)   │         │  + Gripper   │
                        └──────────────┘         └──────────────┘
```

---

## Quick Start

### Terminal 1: Robot Bringup
```bash
source ~/checkers_ws/install/setup.bash
ros2 launch cocohrip_control bringup_all.launch.py
```

### Terminal 2: Robot Control Node
```bash
source ~/checkers_ws/install/setup.bash
ros2 launch robot_control new_launch.py
```

### Terminal 3: Checkers Game (Camera + AI)
```bash
# If needed after reboot (camera memory allocation)
sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0

cd ~/checkers_ws && source install/setup.bash
ros2 launch checkers_game checkers_game_launch.launch.py
```

### Optional: Leap Motion Gesture Interface
```bash
source ~/checkers_ws/install/setup.bash
ros2 run leap_gesture_interface leap_example_node
```

---

## ROS2 Topics Reference

### Published Topics

| Topic | Message Type | Publisher | Description |
|-------|--------------|-----------|-------------|
| `/board_topic` | `checkers_msgs/Board` | checkers_game | Current board state with all pieces |
| `/move_topic` | `checkers_msgs/Move` | checkers_game | AI move command for robot execution |
| `/hand_detected` | `checkers_msgs/HandDetected` | checkers_game | Hand detection over board |
| `/robot_move_topic` | `checkers_msgs/RobotMove` | robot_control | Robot movement completion status |
| `/visualization_marker_array` | `visualization_msgs/MarkerArray` | robot_control | RViz visualization markers |

### Subscribed Topics

| Topic | Message Type | Subscriber | Description |
|-------|--------------|------------|-------------|
| `/move_topic` | `checkers_msgs/Move` | robot_control | Receives move commands |
| `/board_topic` | `checkers_msgs/Board` | robot_control | Board state for visualization |
| `/robot_move_topic` | `checkers_msgs/RobotMove` | checkers_game | Robot completion callback |
| `/hand_detected` | `checkers_msgs/HandDetected` | robot_control | Pause on hand detection |
| `/leap_gesture` | `std_msgs/String` | robot_control | Leap Motion gestures |
| `/leap_gesture/palm_position` | `geometry_msgs/Vector3` | robot_control | Palm position for safety |
| `/joint_states` | `sensor_msgs/JointState` | robot_control | Robot joint feedback |
| `/rosout` | `rcl_interfaces/Log` | robot_control | Movement completion detection |

### Services

| Service | Type | Node | Description |
|---------|------|------|-------------|
| `/resume_movement` | `checkers_msgs/srv/ResumeMovement` | robot_control | Resume paused robot movement |
| `/gripper_service` | `gripper_srv/srv/GripperService` | gripper_driver | Control gripper open/close |

---

## Message Types

### checkers_msgs/Piece.msg
```
int32 row          # Row on board (0-7)
int32 col          # Column on board (0-7)
string color       # "white" or "red" (red = black/AI)
bool king          # Is this piece a king?
```

### checkers_msgs/Board.msg
```
Piece[] pieces     # Array of all pieces on board
```

### checkers_msgs/Move.msg
```
int32 target_row           # Target row for piece movement
int32 target_col           # Target column for piece movement  
Piece piece_for_moving     # The piece to move
Piece[] removed_pieces     # Pieces captured (jumped over)
```

### checkers_msgs/RobotMove.msg
```
bool robot_move_done       # True when robot completes move
```

### checkers_msgs/HandDetected.msg
```
bool hand_detected         # True when hand over board
```

### checkers_msgs/srv/ResumeMovement.srv
```
---
bool success               # Resume operation result
```

---

## Board Coordinates System

### Physical Board Layout

The checkerboard is positioned with:
- **H1** corner closest to camera (start corner for calibration)
- **A1** → **A8** → **H8** (calibration order: counterclockwise)

```
     Camera View                    Robot Coordinate System
    ┌─────────────┐                  
    │ A8       H8 │                 Y ▲
    │             │                   │
    │    Board    │                   │
    │             │                   └───▶ X
    │ A1       H1 │                 Robot Base
    └─────────────┘
      (near robot)
```

### Board Array Coordinates

In code, the board is represented as an 8x8 array:

```
Position = row * 8 + col

       col 0  1  2  3  4  5  6  7
row 0:    0   1   2   3   4   5   6   7    ← BLACK (AI) starts here
row 1:    8   9  10  11  12  13  14  15
row 2:   16  17  18  19  20  21  22  23
row 3:   24  25  26  27  28  29  30  31    ← Empty rows
row 4:   32  33  34  35  36  37  38  39
row 5:   40  41  42  43  44  45  46  47    ← WHITE (Player) starts here
row 6:   48  49  50  51  52  53  54  55
row 7:   56  57  58  59  60  61  62  63
```

### Initial Piece Positions

**BLACK pieces (AI):** Rows 0-2, positions: 1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23

**WHITE pieces (Player):** Rows 5-7, positions: 40, 42, 44, 46, 49, 51, 53, 55, 56, 58, 60, 62

### Coordinate Transformation

The robot control node applies a 180° rotation to match physical board orientation:
```cpp
// RobotControlNode::rotate90DegreesCounterClockwise
int rotated_x = 7 - x;  // Flip X axis  
int rotated_y = 7 - y;  // Flip Y axis
return {y, rotated_x};  // Swap and return
```

---

## Robot Movement Examples

### Debug: Monitor Move Commands
```bash
# Watch what moves the AI is sending to the robot
ros2 topic echo /move_topic
```

**Example output:**
```yaml
target_row: 4
target_col: 3
piece_for_moving:
  row: 2
  col: 5
  color: "red"
  king: false
removed_pieces: []
```

This means: Move the BLACK piece from (2,5) to (4,3).

### Debug: Monitor Robot Status
```bash
# Check when robot finishes movement
ros2 topic echo /robot_move_topic
```

**Example output:**
```yaml
robot_move_done: true
```

### Debug: View Board State
```bash
# See all pieces on the board
ros2 topic echo /board_topic
```

### Manual Move Example

To manually test robot movement (for debugging), publish a move:

```bash
ros2 topic pub /move_topic checkers_msgs/msg/Move "{
  target_row: 3,
  target_col: 2,
  piece_for_moving: {row: 2, col: 1, color: 'red', king: false},
  removed_pieces: []
}" --once
```

### Move with Capture (Removing Opponent's Piece)

```bash
ros2 topic pub /move_topic checkers_msgs/msg/Move "{
  target_row: 4,
  target_col: 3,
  piece_for_moving: {row: 2, col: 5, color: 'red', king: false},
  removed_pieces: [{row: 3, col: 4, color: 'white', king: false}]
}" --once
```

This will:
1. Pick up BLACK piece at (2,5)
2. Place it at (4,3)
3. Pick up WHITE piece at (3,4) - the captured piece
4. Place it in the storage area on the side

### Board placement and storage area (robot_control parameters)

The robot computes every square from two measured points (TCP over the
centre of the dark corner squares, same frame the robot moves in), set in
`robot_control/launch/new_launch.py`:

```
board_h8_xy: [1.1099, 0.4478]   # H8: black piece, corner closest to the robot
board_a1_xy: [0.7191, 0.0458]   # A1: white piece, corner farthest from the robot
z_attach:    0.0035             # fingertip (ur5e_robotiq_hande_end) height when grasping;
                                # launch arg: ros2 launch robot_control new_launch.py z_attach:=0.005
```
The Hand-E gripper is part of the robot model, so MoveIt plans for the
gripper fingertip frame `ur5e_robotiq_hande_end` and collision-checks the
gripper. Its x, y equal the flange's (`ur5e_tool0`); z is
`kHandeFlangeToTcp` = 0.1565 m lower. Read new measurements with
`ros2 run tf2_ros tf2_echo world ur5e_robotiq_hande_end`.
After moving the board, re-measure these two points; a rotated board is
handled (the square size and yaw come from the two points). Also re-check the
camera corners at startup (press `A` in the grid-check window).

Captured / huffed white pieces go beside the **a-file edge**, 2 columns x 6
slots (`storage_first_col` -1.8, `storage_col_step` -1.1 squares,
`storage_rows` 6). The slot positions are printed when robot_control starts.
The slots restart from the first one when a board with 12 white pieces is
published (new game).

### Piece centring and adaptive grip

- **Centring:** the camera measures how far each piece the robot picks sits
  off its square's centre (`Piece.offset_row/offset_col`, in squares). The
  parallax of the piece tops is learned from the start position when you press
  S, so set the start pieces up roughly centred. The robot shifts the pick by
  that offset, at most `pick_offset_max` (0.012 m).
- **Grip:** the gripper closes to `gripper_close_position` (111) +
  `grip_extra_mm` (10 mm, capped at 20) with `grip_force` (20). The Hand-E
  stops on the piece and reports it (OBJ = 2): gripped. If the fingers reach
  the target instead, nothing is between them: it opens and tries again
  (`grip_retries` 1), then gives up the move, goes home and reports
  `RobotMove.pick_failed`. The game then asks you to finish the move by hand
  and continues once the board matches.

---

## Debugging Guide

### Common Issues

#### 1. Robot Not Moving
```bash
# Check if robot_control node is running
ros2 node list | grep robot_control

# Check for movement commands
ros2 topic echo /move_topic

# Check robot status
ros2 topic echo /robot_move_topic
```

#### 2. Detection Issues
```bash
# Check board state being published
ros2 topic echo /board_topic

# Watch for hand detection interrupts
ros2 topic echo /hand_detected
```

#### 3. Resume Paused Movement
```bash
ros2 service call /resume_movement checkers_msgs/srv/ResumeMovement "{}"
```

### Key Debugging Tips

1. **Piece calibration**: at startup put the pieces in the START position and
   press SPACE in the *Piece Calibration* window. Orientation (robot's black
   pieces on top) and the piece model are learned from it - no thresholds to
   tune any more. ESC skips it and uses the built-in defaults.

2. **Board corners**: detected automatically (cached in
   `~/.cocohrip/board_calibration.json`). In the grid-check window: SPACE accept,
   `A` re-detect, `R` click manually (BL, TL, TR, BR on the inner edge of the frame).

3. **During the game** (any OpenCV window focused): `S` start, `F` start from
   the current position, `1/2/3` easy/medium/hard, `SPACE` re-check the board
   now, `R` robot finished (if it did not report), `H` print your legal moves.

4. The `gameboard` window shows what the camera sees per square
   (W / B / `?` occupied-colour-unsure / `??` unknown, red dot = changed since
   the last confirmed position).

---

## Configuration

### Robot Speed Settings

In `RobotControlNode.cpp`:
```cpp
pose_utility_->setVelocityScaling(0.8);    // 80% max velocity
pose_utility_->setAccelerationScaling(0.3); // 30% max acceleration
```

### Movement Heights

Heights are for the gripper fingertip frame `ur5e_robotiq_hande_end`
(flange height minus `kHandeFlangeToTcp` = 0.1565 m):
```cpp
zAttach = 0.160 - kHandeFlangeToTcp;         // 0.0035, ROS param z_attach (grasp)
zMoving = 0.20 - kHandeFlangeToTcp;          // flange at 20 cm
zSafeTransition = 0.25 - kHandeFlangeToTcp;  // flange at 25 cm, travel between squares
```

### AI Difficulty (alpha-beta search)

| Key / trackbar level | Search depth (plies) | |
|------------|-------|-------------|
| `1` EASY = level 1 | 1 | instant, weak |
| level 2 | 2 | |
| `2` MEDIUM = level 3 | 4 | default |
| level 4 | 6 | |
| `3` HARD = level 5 | 8 | capped by `ai_time_limit` (4 s) |

### Detection / game parameters

ROS parameters of `checkers_node`:
```
human_first      true    # false = WCDF order: black (robot) moves first
ai_time_limit    4.0     # seconds per robot move
stable_time      0.8     # board must be still this long before a move is read
motion_threshold 1.5     # % of changed pixels that counts as motion
```

---

## System Flow

```
1. S pressed with the start position -> you (white) move first
2. Human moves a white piece and takes the hand away; when the detected
   board has been still for stable_time the move is matched against every
   legal move (and every huff-able one)
3. Robot turn: if the human ignored a capture, the robot first "huffs"
   (removes) one of the pieces that should have captured; then the AI picks
   a legal move (captures are mandatory, multi-jumps completed)
4. /move_topic: removed_pieces = huffed piece + captured pieces
5. robot_control removes those pieces FIRST, then moves its own piece
   (a huffed piece can stand on the target square)
6. robot_move_done=true is published once, after the last pick/place
7. checkers_game checks that the board now matches the expected position
8. Repeat until a side has no pieces / no legal move, or a draw
```

---

## Checkers game package (rules, vision, tools)

### Rules - English draughts / American checkers (WCDF)
- Men move one square diagonally forward, capture forward only.
- Kings move/capture one square diagonally in any direction (no flying kings).
- **Capturing is mandatory**; with several captures any one may be chosen.
- A multi-jump must be completed; a piece cannot be jumped twice.
- A man reaching the last row is crowned and **the move ends** there.
- The human (white) moves first by default (`human_first:=false` gives the
  WCDF order, black/robot first). No pieces or no legal move = loss.
- Draw: 40 moves each without capture or man move, or 3-fold repetition.
- **Huff rule**: if the human plays a non-capturing move while a capture was
  available (or stops a multi-jump early) the move stands, but the robot
  removes one of the pieces that should have captured, then makes its move.
  Captures the human makes: the human takes the captured robot pieces off.

Code: `checkers/rules.py` (engine), `checkers/game.py` (game flow, move
inference from the camera), `minimax/algorithm.py` (alpha-beta AI).

### The game window
One window ("Checkers", resizable) during the game: the game board (the
position the game believes in, last move in blue, problem squares red, small
dots = what the camera sees on each square), the camera with the board
outline and hand tracking, the rectified detection view, and a panel with
status, how still the board is, exposure, the move list and the keys.
The startup calibration steps still use their own windows.

### Vision
- `camera/board_geometry.py` - board corners from the squares' lattice.
- `camera/board_vision.py` - rectification into game orientation, lighting
  invariant per-square features (reflectance relative to the square's own
  border + a circular-rim matched filter that finds even plain black pieces
  on black squares), piece classifier, change detection.
- `camera/board_detection.py` - the calibration windows and runtime fusion:
  classifier + "did this square change since the last confirmed position".
- `camera/ximea_camera.py` - software auto-exposure on the board's white
  squares (startup, `E` key, and automatic when the lamp is switched on/off).

Measured on recordings from the cell (2 lightings x 3 exposures): 99.2-100 %
of dark squares classified correctly *without any calibration*, no wrong
labels (the rest is reported as "unsure" and resolved by the rules).

### Tools (no ROS needed)
```bash
# record a dataset on the robot PC (stop the game node first)
python3 checkers_game/tools/capture_dataset.py
# evaluate detection on it / replay a recorded game
python3 checkers_game/tools/eval_detection.py ~/checkers_captures/<stamp>
python3 checkers_game/tools/replay_video.py ~/checkers_captures/<stamp>/video
# tests (rules, game flow, vision on real frames, simulated full node run)
cd checkers_game && python3 -m pytest test/test_rules.py test/test_game.py \
    test/test_board_vision.py test/test_real_frames.py test/test_board_geometry.py \
    test/test_node_sim.py -q
```

---

## Authors & License

Cocohrip Checkers Robot System - AMAVET Robotics Project

For issues or contributions, please contact the project maintainers.
