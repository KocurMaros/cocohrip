# URDF changes

Simulation and real robot use the **same** URDF; only the launch argument differs
(`use_mock_hardware:=true` for simulation, `robot_ip:=<ip>` for the real UR5e).

## Where to edit

| File | What it holds |
|------|---------------|
| `cocohrip_description/urdf/cocohrip_cell_macro.xacro` | The cell: table, robot mount pose, UR5e, gripper. **Put geometry changes here.** |
| `robotiq_hande_description/urdf/robotiq_hande_gripper.xacro` | Gripper links/joints and the TCP frame `robotiq_hande_end`. |
| `cocohrip_control/urdf/cocohrip_controlled.urdf.xacro` | Top-level file for `robot_state_publisher` + `ros2_control` (sim and real). Add `ros2_control` tags here. |
| `cocohrip_description/urdf/cocohrip_cell.urdf.xacro` | Top-level file for MoveIt (planning, RViz). |
| `cocohrip_moveit_config/config/cocohrip_cell.srdf` | Planning groups and the collision matrix. |
| `cocohrip_control/config/ur5e_calibration.yaml` | Factory calibration of the real arm. |

## Rules

1. Edit the macro, not the top-level files. Both top-level files include it, so the
   simulation, the real robot and MoveIt stay in sync. A new xacro argument must be added
   to **both** top-level files.
2. Meshes: keep visual meshes light (e.g. `cocohrip_visual.dae`) and use primitives or a
   coarse mesh for collision. Heavy visual meshes slow down RViz.
3. New or renamed links: update the SRDF groups and regenerate `disable_collisions`
   (`ros2 launch cocohrip_moveit_config setup_assistant.launch.py` → *Self-Collisions*).
4. Changed gripper or TCP: `robot_control` plans for `ur5e_robotiq_hande_end`; update
   `kHandeFlangeToTcp` in `RobotControlNode.hpp` and the `z_attach` launch parameter.
5. New real arm: regenerate the calibration file
   (`ros2 launch ur_calibration calibration_correction.launch.py robot_ip:=<ip>
   target_filename:=<path>/ur5e_calibration.yaml`). MoveIt's top-level file uses the UR
   default kinematics, not this file.

## Check

```bash
colcon build --packages-select cocohrip_description cocohrip_control cocohrip_moveit_config
source install/setup.bash
xacro $(ros2 pkg prefix --share cocohrip_description)/urdf/cocohrip_cell.urdf.xacro > /tmp/cell.urdf
check_urdf /tmp/cell.urdf
ros2 launch cocohrip_control bringup_all.launch.py use_mock_hardware:=true   # simulation
```
