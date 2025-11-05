#include "robot_control/RobotControlNode.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    // Create the node as a shared pointer
    auto robot_control_node = std::make_shared<RobotControlNode>();
    
    // Initialize move group AFTER the node is fully constructed
    robot_control_node->initMoveGroup();
    
    // Create executor with timer for main loop
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(robot_control_node);
    
    // Create a timer that calls mainLoop periodically
    auto timer = robot_control_node->create_wall_timer(
        std::chrono::milliseconds(100),
        [&robot_control_node]() {
            robot_control_node->mainLoop();
        }
    );
    
    RCLCPP_INFO(robot_control_node->get_logger(), "Robot Control Node started successfully!");
    
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}