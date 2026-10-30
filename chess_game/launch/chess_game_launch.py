from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        # stockfish | leela | maia (level = Maia 1100..1900) | lc0:weights=... | uci:/path | builtin
        DeclareLaunchArgument('engine', default_value='stockfish'),
        DeclareLaunchArgument('level', default_value='3'),
        DeclareLaunchArgument('model_path',
                              default_value='~/chess_models/latest/piece_classifier.onnx'),
        DeclareLaunchArgument('colour_only', default_value='false'),
        # '' = share/chess_game/config/pieces.yaml (the cell's wooden set)
        DeclareLaunchArgument('pieces_yaml', default_value=''),
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
                # True: classifier used for empty / white / black only (new piece set)
                'colour_only': ParameterValue(LaunchConfiguration('colour_only'), value_type=bool),
                'pieces_yaml': ParameterValue(LaunchConfiguration('pieces_yaml'), value_type=str),
                'ai_time_limit': 2.0,     # s per robot move
                'stable_time': 0.8,       # s the board must be still before it is read
                # storage for captured pieces: one column left, one right of the board
                'storage_left_col': -1.8,
                'storage_right_col': 8.8,
                'storage_rows': 8,
            }],
        ),
    ])
