cmake_minimum_required(VERSION 3.8)
project(demo_node)

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
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(checkers_msgs REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(rviz_visual_tools REQUIRED)
find_package(gripper_srv REQUIRED)

# Include C++ headers
include_directories(include)

# Add simple pose reader executable
add_executable(simple_pose_reader src/SimplePoseReader.cpp)
ament_target_dependencies(simple_pose_reader
        moveit_core
        moveit_ros_planning_interface
        rclcpp
        geometry_msgs
)
target_include_directories(simple_pose_reader PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# Install the executable
install(TARGETS
        simple_pose_reader
        DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
)

# Install etc files
install(DIRECTORY etc
        DESTINATION share/${PROJECT_NAME}
)

# Install urdf files
install(DIRECTORY urdf
        DESTINATION share/${PROJECT_NAME}
)

# Install rviz files
install(DIRECTORY rviz
        DESTINATION share/${PROJECT_NAME}
)


if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()