#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    # Initialize Arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

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
                #   H8 = black piece, corner closest to the robot
                #   A1 = white piece, corner farthest from the robot
                # Re-measure both after moving the board (rotation is handled).
                "board_h8_xy": [1.1099, 0.4478],
                "board_a1_xy": [0.7191, 0.0458],
                "z_attach": 0.155,           # grasp height of the TCP
                # Captured / huffed pieces: beside the a-file edge, slot k at
                # game row k % storage_rows, column first + (k // rows) * step
                # (in squares; column 0 = a-file, negative = outside the board)
                "storage_first_col": -1.8,
                "storage_col_step": -1.1,
                "storage_rows": 6,
            },
        ],
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True,
    )

    nodes_to_start = [
        robot_control_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)