cmake_minimum_required(VERSION 3.8)
project(robot_control)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rviz2 REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(checkers_msgs REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(rviz_visual_tools REQUIRED)
find_package(gripper_srv REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(leap_gesture_interface REQUIRED)

# Include C++ headers
include_directories(include)

# ============================================================
# 1. RobotPoseUtility Library (shared utility)
# ============================================================
add_library(robot_pose_utility src/RobotPoseUtility.cpp)
ament_target_dependencies(robot_pose_utility
    moveit_core
    moveit_ros_planning_interface
    rclcpp
    geometry_msgs
    sensor_msgs
    leap_gesture_interface
)
target_include_directories(robot_pose_utility PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ============================================================
# 2. Main robot_control_node executable
# ============================================================
add_executable(robot_control_node 
    src/RobotControlMain.cpp 
    src/RobotControlNode.cpp 
    src/Mission.cpp
)

# Link against robot_pose_utility library
target_link_libraries(robot_control_node robot_pose_utility)

ament_target_dependencies(robot_control_node
    moveit_core
    moveit_ros_planning_interface
    rclcpp
    visualization_msgs
    geometry_msgs
    tf2
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
    checkers_msgs
    moveit_visual_tools
    gripper_srv
    rviz_visual_tools
    leap_gesture_interface
)
target_include_directories(robot_control_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ============================================================
# 3. SimplePoseReader standalone executable
# ============================================================
add_executable(simple_pose_reader src/SimplePoseReader.cpp)
ament_target_dependencies(simple_pose_reader
    moveit_core
    moveit_ros_planning_interface
    rclcpp
    geometry_msgs
    sensor_msgs
)
target_include_directories(simple_pose_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ============================================================
# 4. TestRobotPoseUtility executable (basic test)
# ============================================================
add_executable(test_robot_pose_utility src/TestRobotPoseUtility.cpp)
target_link_libraries(test_robot_pose_utility robot_pose_utility)
ament_target_dependencies(test_robot_pose_utility
    rclcpp
    geometry_msgs
)

# ============================================================
# 5. TestRobotPoseUtilityComprehensive (comprehensive test suite)
# ============================================================
add_executable(test_robot_pose_utility_comprehensive src/TestRobotPoseUtilityComprehensive.cpp)
target_link_libraries(test_robot_pose_utility_comprehensive robot_pose_utility)
ament_target_dependencies(test_robot_pose_utility_comprehensive
    rclcpp
    geometry_msgs
)

# ============================================================
# 6. TestMovePublisher executable
# ============================================================
add_executable(test_move_publisher 
    src/TestMovePublisher.cpp
    src/TestMovePublisherMain.cpp
)
ament_target_dependencies(test_move_publisher
    rclcpp
    checkers_msgs
)
target_include_directories(test_move_publisher PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ============================================================
# Add direct movement test executable
# ============================================================
add_executable(test_direct_move src/TestDirectMove.cpp)
ament_target_dependencies(test_direct_move ${dependencies})
target_link_libraries(test_direct_move robot_pose_utility)

# ============================================================
# Install targets
# ============================================================

# Install library
install(TARGETS robot_pose_utility
    EXPORT robot_pose_utilityTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install executables
install(TARGETS
    robot_control_node
    simple_pose_reader
    test_robot_pose_utility
    test_robot_pose_utility_comprehensive
    test_move_publisher
    test_direct_move
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and rviz configs
install(DIRECTORY
    launch
    rviz
    DESTINATION share/${PROJECT_NAME}/
)

# Install header files
install(DIRECTORY include/
    DESTINATION include/
)

# Install URDF and config files
install(DIRECTORY etc urdf 
    DESTINATION share/${PROJECT_NAME}
)

# ============================================================
# Testing
# ============================================================
if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
