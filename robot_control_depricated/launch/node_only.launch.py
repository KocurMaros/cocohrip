#!/usr/bin/env python3
"""
Launch file that only starts the move_ur5e_node.
Use this when UR driver and MoveIt are already running.
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    # Robot control node - delayed start to ensure MoveGroup is ready
    robot_control_node = Node(
        package="robot_control_amavet",
        executable="move_ur5e_node",
        name="move_ur5e_amavet",
        output="screen",
        parameters=[
            {"use_sim_time": False},
        ],
    )
    
    # Delay node start by 2 seconds to allow MoveGroup to initialize
    delay_node = TimerAction(
        period=2.0,
        actions=[robot_control_node],
    )
    
    return LaunchDescription([delay_node])
