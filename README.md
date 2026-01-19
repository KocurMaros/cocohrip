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
source ~/amavet_ws/install/setup.bash
ros2 launch cocohrip_control bringup_all.launch.py
```

### Terminal 2: Robot Control Node
```bash
source ~/amavet_ws/install/setup.bash
ros2 launch robot_control new_launch.py
```

### Terminal 3: Checkers Game (Camera + AI)
```bash
# If needed after reboot (camera memory allocation)
sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0

cd ~/amavet_ws && source install/setup.bash
ros2 launch checkers_game checkers_game_launch.launch.py
```

### Optional: Leap Motion Gesture Interface
```bash
source ~/amavet_ws/install/setup.bash
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

### Storage Area for Captured Pieces

Captured pieces are stored in columns of 4 pieces each:

```
Storage Layout (beside the board):
Column 0: pieces 0-3
Column 1: pieces 4-7  
Column 2: pieces 8-11

Physical location:
  row = 9 + (piece_number % 4)
  col = piece_number / 4
```

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

1. **Variance Thresholds**: During calibration, set thresholds to clearly separate:
   - Empty squares: variance < 600
   - Black pieces: 600 < variance < 1000
   - White pieces: variance > 1000

2. **Board Corners**: Select corners in order: H1 → A1 → A8 → H8

3. **Difficulty Levels**: Press keys during game:
   - `1` = EASY (depth 1)
   - `2` = MEDIUM (depth 3) 
   - `3` = HARD (depth 5)

4. **Force Move Validation**: Press `SPACE` to force-validate current board state

---

## Configuration

### Robot Speed Settings

In `RobotControlNode.cpp`:
```cpp
pose_utility_->setVelocityScaling(0.8);    // 80% max velocity
pose_utility_->setAccelerationScaling(0.3); // 30% max acceleration
```

### Movement Heights

```cpp
zAttach = 0.155;        // Height for gripping pieces
zMoving = 0.20;         // Safe movement height (20cm above board)
zSafeTransition = 0.25; // Height for transitions between positions
```

### AI Difficulty (minimax depth)

| Difficulty | Depth | Performance |
|------------|-------|-------------|
| EASY | 1 | Very fast, makes mistakes |
| MEDIUM | 3 | Balanced (default) |
| HARD | 5 | Slow but very strong |

### Detection Parameters

In `BoardDetection`:
```python
STABILITY_THRESHOLD = 0.8  # Seconds to wait for stable board
HAND_COOLDOWN = 0.5        # Seconds after hand leaves before detection
```

---

## System Flow

```
1. Game starts → Player makes move → Camera detects change
2. Board stabilizes → Move validated → Turn changes to AI
3. AI calculates best move (minimax) → Publishes to /move_topic
4. robot_control receives move → Plans trajectory
5. Robot executes: 
   a) Moves to safe approach position (20cm above board center)
   b) Picks up piece
   c) Places piece at target
   d) If capture: picks up captured piece → stores in side column
6. Robot publishes robot_move_done=true to /robot_move_topic
7. checkers_game receives callback → Changes turn to Player
8. Repeat from step 1
```

---

## Authors & License

Cocohrip Checkers Robot System - AMAVET Robotics Project

For issues or contributions, please contact the project maintainers.
