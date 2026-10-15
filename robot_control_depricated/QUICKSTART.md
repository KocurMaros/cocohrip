# Quick Start Guide - Robot Control Amavet

## Current Issue: RTDE Connection Error

Your error shows:
```
Failed to initialize RTDE client
Variable 'speed_slider_mask' is currently controlled by another RTDE client
```

**This means another program is already connected to the robot.**

## Solutions

### Option 1: Disconnect Other Clients

1. **Check the teach pendant** - Is PolyScope running a program?
2. **Check for URSim** - Close any URSim instances
3. **Check for other ROS drivers** - Stop any other terminals running `ur_control.launch.py`

### Option 2: Use Fake Hardware (Simulation)

If you don't need the real robot right now:

```bash
cd /home/collab/amavet_ws
source install/setup.bash
ros2 launch robot_control_amavet amavet_control.launch.py use_fake_hardware:=true
```

This will start everything in simulation mode without connecting to real hardware.

### Option 3: Run Node Only (if driver is already running elsewhere)

If you already have the UR driver running in another terminal:

```bash
cd /home/collab/amavet_ws
source install/setup.bash
ros2 launch robot_control_amavet node_only.launch.py
```

Or run the node directly:

```bash
ros2 run robot_control_amavet move_ur5e_node
```

## What the Node Does Now

The updated node is more robust:

1. ✅ **Gracefully handles missing TF frames** - Warns instead of aborting
2. ✅ **Waits for MoveGroup to be ready** - Better initialization
3. ✅ **Continues running even if hardware not connected** - Useful for development
4. ✅ **Better error messages** - Easier to debug

## Recommended Workflow

### For Real Robot:

1. **First, make sure robot is accessible:**
   ```bash
   ping 192.168.1.102
   ```

2. **Make sure no other programs are connected to the robot**

3. **Launch everything:**
   ```bash
   ros2 launch robot_control_amavet amavet_control.launch.py robot_ip:=192.168.1.102
   ```

### For Development/Testing:

1. **Use fake hardware:**
   ```bash
   ros2 launch robot_control_amavet amavet_control.launch.py use_fake_hardware:=true
   ```

2. **This gives you:**
   - Full MoveIt functionality
   - No real hardware needed
   - Can test motion planning
   - Can develop/debug code safely

## Checking System Status

```bash
# Check if UR driver is running
ros2 node list | grep ur

# Check if MoveGroup is running
ros2 node list | grep move_group

# Check if robot_description is published
ros2 topic list | grep robot_description

# Check TF frames
ros2 run tf2_tools view_frames

# Check controllers
ros2 control list_controllers
```

## Next Steps

Once you resolve the RTDE connection issue, the node will:
1. Initialize MoveItPy ✅
2. Get current robot pose ✅
3. Perform a test move (10cm up) ✅
4. Test gripper (close then open) ✅
5. Run main control loop (ready for your code) ✅

The package is ready to use - just need to resolve the robot connection!
