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
            {"use_sim_time": use_sim_time}
        ],
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True,
    )

    nodes_to_start = [
        robot_control_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)