import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, TimerAction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import yaml
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_subpath):
    """Load YAML file from package."""
    pkg_share = get_package_share_directory(package_name)
    full_path = os.path.join(pkg_share, file_subpath)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def launch_setup(context, *args, **kwargs):
    # Launch configuration
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    launch_driver = LaunchConfiguration("launch_driver")
    
    # Launch UR driver if needed
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ur_robot_driver"),
                "launch",
                "ur_control.launch.py"
            ])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": "false",
        }.items(),
        condition=IfCondition(launch_driver)
    )
    
    # Launch MoveIt
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ur_moveit_config"),
                "launch",
                "ur_moveit.launch.py"
            ])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": "false",
        }.items(),
    )
    
    # Robot control node - delayed start
    robot_control_node = Node(
        package="robot_control_amavet",
        executable="move_ur5e_node",
        name="move_ur5e_amavet",
        output="screen",
        parameters=[
            {"use_sim_time": False},
        ],
    )
    
    delay_node = TimerAction(
        period=5.0,
        actions=[robot_control_node],
    )
    
    nodes_to_start = [ur_control_launch, moveit_launch, delay_node]
    return nodes_to_start


def generate_launch_description():
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type of UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"],
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.102",
            description="IP address of the robot.",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use fake hardware for testing.",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_driver",
            default_value="true",
            description="Launch the UR robot driver.",
        )
    )
    
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])