from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        # stockfish | lc0:weights=/path/maia-1500.pb.gz | builtin | random
        DeclareLaunchArgument('engine', default_value='stockfish'),
        DeclareLaunchArgument('level', default_value='3'),
        DeclareLaunchArgument('model_path',
                              default_value='~/chess_models/latest/piece_classifier.onnx'),
    ]
    return LaunchDescription(args + [
        Node(
            package='chess_game',
            executable='chess_node',
            name='chess_node',
            output='screen',
            parameters=[{
                'engine': LaunchConfiguration('engine'),
                'level': ParameterValue(LaunchConfiguration('level'), value_type=int),
                'model_path': LaunchConfiguration('model_path'),
                'ai_time_limit': 2.0,     # s per robot move
                'stable_time': 0.8,       # s the board must be still before it is read
                'pieces_yaml': '',        # '' = share/chess_game/config/pieces.yaml
                # storage for captured pieces: one column left, one right of the board
                'storage_left_col': -1.8,
                'storage_right_col': 8.8,
                'storage_rows': 8,
            }],
        ),
    ])
