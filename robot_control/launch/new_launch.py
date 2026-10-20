#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """
    Launch file for standalone robot_control_node.
    
    This assumes that ur_control and MoveIt are already running in other terminals.
    This node will connect to the existing move_group and robot_description.
    """
    
    declared_arguments = []
    
    # Declare launch arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time (set to true if using Gazebo)"
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="Logging level (debug, info, warn, error, fatal)",
            choices=["debug", "info", "warn", "error", "fatal"]
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "z_attach",
            # 0.0035 = flange at 0.160 m, the height the real robot actually grasped
            # at before the gripper was in the model (the old 0.155 target was
            # clamped to RobotPoseUtility's 0.16 m minimum flange height).
            default_value="0.0035",
            description="Gripper fingertip (ur5e_robotiq_hande_end) height in meters when grasping a piece",
        )
    )

    # Initialize Arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    z_attach = LaunchConfiguration("z_attach")

    # Robot Control Node
    robot_control_node = Node(
        package="robot_control",
        executable="robot_control_node",
        name="robot_control_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {
                # Board placement: TCP (x, y) over the centre of the two dark
                # corner squares, read in the same frame the robot moves in.
                # (Measured on the flange ur5e_tool0 before the gripper was in
                # the model; x, y carry over unchanged to the fingertip frame
                # ur5e_robotiq_hande_end, which sits straight below the flange.)
                #   H8 = black piece, corner closest to the robot
                #   A1 = white piece, corner farthest from the robot
                # Re-measure both after moving the board (rotation is handled).
                "board_h8_xy": [1.1099, 0.4478],
                "board_a1_xy": [0.7191, 0.0458],
                # Captured / huffed pieces: beside the a-file edge, slot k at
                # game row k % storage_rows, column first + (k // rows) * step
                # (in squares; column 0 = a-file, negative = outside the board)
                "storage_first_col": -1.8,
                "storage_col_step": -1.1,
                "storage_rows": 6,
            },
            # grasp height of the gripper fingertip frame (ur5e_robotiq_hande_end)
            {"z_attach": ParameterValue(z_attach, value_type=float)},
        ],
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True,
    )

    nodes_to_start = [
        robot_control_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)