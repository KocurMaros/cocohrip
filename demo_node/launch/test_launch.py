import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type/series of used UR robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.0.5",
            description="IP address of the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Use mock hardware instead of real robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch RViz.",
        )
    )

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")

    # 1. Launch UR robot driver
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"])]
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "use_mock_hardware": use_mock_hardware,
            "launch_rviz": "false",
        }.items(),
    )

    # 2. Launch MoveIt
    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py"])]
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
            "use_sim_time": "false",
        }.items(),
    )

    # 3. Simple pose reader node
    simple_pose_node = Node(
        package="demo_node",
        executable="simple_pose_reader",
        name="simple_pose_reader",
        output="screen",
    )

    # Delay the pose reader to ensure everything is ready
    delay_pose_reader = TimerAction(
        period=15.0,  # 15 seconds delay
        actions=[simple_pose_node],
    )

    nodes_to_start = [
        ur_control_launch,
        ur_moveit_launch,
        delay_pose_reader,
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)