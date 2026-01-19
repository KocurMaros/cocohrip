#include <rclcpp/rclcpp.hpp>
#include "robot_control/RobotPoseUtility.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<rclcpp::Node>("test_direct_move");
    
    RCLCPP_INFO(node->get_logger(), "Initializing RobotPoseUtility for chessboard loop test...");
    
    auto robot_util = std::make_shared<RobotPoseUtility>(
        node,
        "ur5e_arm",
        "ur5e_tool0"
    );
    
    if (!robot_util->initialize()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize RobotPoseUtility!");
        return 1;
    }
    
    // Real-world measured board corners (center of corner squares)
    // Closest to robot (bottom-right corner)
    float corner_near_x = 0.8736;
    float corner_near_y = 0.4882+0.01;
    
    // Farthest from robot (top-left corner)
    float corner_far_x = 0.4711;
    float corner_far_y = 0.0866+0.01;
    
    // Calculate other two corners
    // Bottom-left corner (near Y, far X)
    float corner_bottom_left_x = corner_far_x;
    float corner_bottom_left_y = corner_near_y;
    
    // Top-right corner (far Y, near X)
    float corner_top_right_x = corner_near_x;
    float corner_top_right_y = corner_far_y;
    
    float zMoving = 0.19;  // Safe height above board
    
    // Orientation pointing down (from RobotControlNode)
    geometry_msgs::msg::Pose corner_pose;
    corner_pose.orientation.x = -0.0028119066264480352;
    corner_pose.orientation.y = 0.9999957084655762;
    corner_pose.orientation.z = -0.0007648332393728197;
    corner_pose.orientation.w = -0.00023792324645910412;
    corner_pose.position.z = zMoving;
    
    RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║  Moving Robot Around Chessboard Perimeter        ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(node->get_logger(), "\nBoard corners (center of corner squares):");
    RCLCPP_INFO(node->get_logger(), "  Bottom-Left:  X: %.4f, Y: %.4f", corner_bottom_left_x, corner_bottom_left_y);
    RCLCPP_INFO(node->get_logger(), "  Bottom-Right: X: %.4f, Y: %.4f", corner_near_x, corner_near_y);
    RCLCPP_INFO(node->get_logger(), "  Top-Left:     X: %.4f, Y: %.4f", corner_far_x, corner_far_y);
    RCLCPP_INFO(node->get_logger(), "  Top-Right:    X: %.4f, Y: %.4f", corner_top_right_x, corner_top_right_y);
    RCLCPP_INFO(node->get_logger(), "  Z (moving):   %.4f", zMoving);
    
    // Define corner positions (starting from closest corner, going counter-clockwise)
    std::vector<std::pair<float, float>> corners = {
        {corner_bottom_left_x, corner_bottom_left_y}, // Corner 2: Bottom-Left (0,7)
        {corner_far_x, corner_far_y},                // Corner 3: Top-Left (0,0) - Farthest from robot
        {corner_top_right_x, corner_top_right_y},    // Corner 4: Top-Right (7,0)
        {corner_near_x, corner_near_y},               // Back to Corner 1 to complete loop
        {corner_bottom_left_x, corner_bottom_left_y}, // Corner 2: Bottom-Left (0,7)
        
    };
    
    const char* corner_names[] = {
        "Bottom-Right (7,7) - Closest",
        "Bottom-Left (0,7)",
        "Top-Left (0,0) - Farthest",
        "Top-Right (7,0)",
        "Bottom-Right (7,7) - Loop Complete"
    };
    
    // Move to each corner in sequence
    for (size_t i = 0; i < corners.size(); ++i) {
        corner_pose.position.x = corners[i].first;
        corner_pose.position.y = corners[i].second;
        
        RCLCPP_INFO(node->get_logger(), "\n→ Moving to %s", corner_names[i]);
        RCLCPP_INFO(node->get_logger(), "   Position: [X: %.4f, Y: %.4f, Z: %.4f]",
                    corner_pose.position.x,
                    corner_pose.position.y,
                    corner_pose.position.z);
        
        if (robot_util->moveToPose(corner_pose)) {
            RCLCPP_INFO(node->get_logger(), "✅ Reached %s", corner_names[i]);
            std::this_thread::sleep_for(2s);  // Pause at each corner
        } else {
            RCLCPP_ERROR(node->get_logger(), "❌ Failed to reach %s!", corner_names[i]);
            RCLCPP_ERROR(node->get_logger(), "Aborting loop test.");
            rclcpp::shutdown();
            return 1;
        }
    }
    
    RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║  ✅ Successfully completed chessboard loop!       ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(node->get_logger(), "\nRobot is now at starting position.");
    RCLCPP_INFO(node->get_logger(), "Press Ctrl+C to exit.");
    
    // Keep node alive to hold position
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}