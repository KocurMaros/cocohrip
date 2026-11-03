cmake_minimum_required(VERSION 3.8)
project(chess_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PickPlace.msg"
  "msg/ChessRobotMove.msg"
  "msg/ChessState.msg"
)

ament_package()
