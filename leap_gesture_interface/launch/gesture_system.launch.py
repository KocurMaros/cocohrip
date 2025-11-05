from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    namespace = 'leap_motion'

    return LaunchDescription([
        # Leap Motion hand publisher node
        Node(
            package='leap_gesture_interface',
            executable='leap_hand_publisher_node',
            name='leap_hand_publisher_node',
            output='screen',
            parameters=[]  # Add any parameters here if needed
        ),

        # Leap Motion pointing intersection publisher node
        Node(
            package='leap_gesture_interface',
            executable='pointing_intersection_publisher',
            name='pointing_intersection_publisher',
            output='screen',
            parameters=[],
            arguments=['--ros-args', '--log-level', 'info']
        ),

        # Leap Motion gesture interface node
        Node(
            package='leap_gesture_interface',
            executable='gesture_interface_node',
            name='gesture_interface_node',
            output='screen',
            parameters=[
                {'filter_frames': 5},
                {'init_gesture': 'FIST'}
            ],
            arguments=['--ros-args', '--log-level', 'info']
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_world_to_leap_hand',
            output='screen',
            arguments=[
                # Translation: x y z (in meters)
                '0.0', '0.0', '0.0',
                # Rotation: roll pitch yaw (in radians)
                '0', '0', '0',
                # Frames: parent → child
                'world',          # parent_frame
                'leap_hand'       # child_frame
            ]
        )
    ])