# Robot Control Amavet Package

A Python-based ROS 2 package for controlling UR5e robot with MoveIt and gripper integration.

## Overview

This package provides:
- **MoveItPy** integration for motion planning
- **TF2** for pose tracking and transforms
- **Gripper service** client for gripper control
- **Launch file** that starts UR driver and MoveIt (without RViz)

## Package Structure

```
robot_control_amavet/
├── robot_control_amavet/
│   ├── __init__.py
│   └── move_ur5e_node.py          # Main robot control node
├── launch/
│   └── amavet_control.launch.py   # Launch file
├── resource/
├── test/
├── package.xml                     # Package dependencies
├── setup.py                        # Python package setup
└── setup.cfg
```

## Dependencies

- `rclpy` - ROS 2 Python client library
- `moveit` - MoveItPy for motion planning
- `geometry_msgs` - For Pose messages
- `tf2_ros` - For transforms
- `gripper_srv` - Custom gripper service
- `ur_robot_driver` - UR robot driver
- `ur_moveit_config` - MoveIt configuration for UR robots

## Building

```bash
cd /home/collab/amavet_ws
colcon build --packages-select robot_control_amavet
source install/setup.bash
```

## Usage

### Launch with Real Robot

```bash
ros2 launch robot_control_amavet amavet_control.launch.py robot_ip:=192.168.1.102
```

### Launch with Fake Hardware (Simulation)

```bash
ros2 launch robot_control_amavet amavet_control.launch.py use_fake_hardware:=true
```

### Launch Only the Control Node (if driver and MoveIt are already running)

```bash
ros2 launch robot_control_amavet amavet_control.launch.py launch_driver:=false
```

### Run Node Directly

```bash
ros2 run robot_control_amavet move_ur5e_node
```

## Launch Arguments

- `ur_type` (default: "ur5e") - Type of UR robot
- `robot_ip` (default: "192.168.1.102") - IP address of the robot
- `use_fake_hardware` (default: "false") - Use fake hardware for testing
- `launch_driver` (default: "true") - Launch the UR robot driver

## Node Features

### Initialization Sequence

The node performs the following on startup:
1. Waits 3 seconds for TF buffer to populate
2. Gets current end-effector pose
3. Initializes MoveItPy planning component
4. Performs a test move (moves 10cm up)
5. Tests gripper (close then open)

### Main Functionality

- **move_to_pose(pose)**: Plans and executes motion to target pose
- **get_current_pose()**: Gets current end-effector pose via TF2
- **open_gripper()**: Opens gripper via service call
- **close_gripper()**: Closes gripper via service call
- **main_loop()**: Main control loop (currently empty, ready for implementation)

### Configuration

Board parameters (for checkers game integration):
- `square_size`: 0.055m
- `board_offset_x`: Calculated based on board size
- `board_offset_y`: Calculated based on board position

## API Differences from ROS 1

This package uses **MoveItPy** (ROS 2) instead of **moveit_commander** (ROS 1):

| ROS 1 (moveit_commander) | ROS 2 (MoveItPy) |
|--------------------------|------------------|
| `moveit_commander.RobotCommander()` | `MoveItPy()` |
| `moveit_commander.MoveGroupCommander("group")` | `PlanningComponent("group", moveit)` |
| `move_group.set_pose_target(pose)` | `planning_component.set_goal_state(pose_stamped_msg=pose)` |
| `move_group.plan()` | `planning_component.plan()` |
| `move_group.execute(plan)` | `moveit.execute(plan_result.trajectory)` |

## Troubleshooting

### ModuleNotFoundError: No module named 'moveit_commander'

This is expected - ROS 2 uses `moveit` (MoveItPy) instead of `moveit_commander`.

### RTDE Error: "Variable 'speed_slider_mask' is currently controlled by another RTDE client"

This means another program is already connected to the robot's RTDE interface. To fix:

1. **Check for other connections:**
   ```bash
   # Check if PolyScope is running on the teach pendant
   # Close any URSim instances
   # Stop any other ROS drivers
   ```

2. **Restart the robot controller:**
   - Power cycle the robot or restart the URSim
   - Make sure no other programs are using the RTDE interface

3. **Check robot IP and connectivity:**
   ```bash
   ping 192.168.1.102
   ```

### TF lookup failed / "base_link does not exist"

This is normal on startup if the robot hardware is not connected. The node will continue running and retry when needed. To fix:

1. Make sure the robot is properly connected and the UR driver is running
2. Check that `/robot_description` is being published:
   ```bash
   ros2 topic echo /robot_description --once
   ```
3. Increase the initialization delay if needed (already set to 5 seconds)

### Gripper service unavailable

Make sure the gripper service node is running:
```bash
ros2 service list | grep gripper
```

### MoveGroup not available

The node now gracefully handles this - it will wait for MoveGroup to start. Make sure:
```bash
ros2 node list | grep move_group
```

## Future Enhancements

- Implement checkers game move callbacks
- Add collision object management
- Implement trajectory generation for missions
- Add RViz markers for visualization

## License

TODO: Add license information

## Maintainer

- Amavet
