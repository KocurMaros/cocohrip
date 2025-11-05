cmake_minimum_required(VERSION 3.8)
project(cocohrip_moveit_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(gripper_srv REQUIRED)

# Include headers
include_directories(include)

add_executable(key_input_publisher src/key_input_publisher.cpp)
ament_target_dependencies(key_input_publisher rclcpp std_msgs)
install(TARGETS key_input_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(ur_position_control_node src/ur_position_control_node.cpp src/ur_position_control.cpp)
add_executable(ur_scenarios_node src/scenarios_node.cpp)

ament_target_dependencies(ur_position_control_node rclcpp std_msgs geometry_msgs moveit_core moveit_ros_planning_interface gripper_srv)
ament_target_dependencies(ur_scenarios_node rclcpp std_msgs geometry_msgs moveit_core moveit_ros_planning_interface gripper_srv)

target_include_directories(ur_scenarios_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
install(TARGETS ur_scenarios_node
  DESTINATION lib/${PROJECT_NAME})

target_include_directories(ur_position_control_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS ur_scenarios_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS ur_position_control_node
  DESTINATION lib/${PROJECT_NAME})
  
install(DIRECTORY include/ launch/
  DESTINATION include)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  srdf
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
