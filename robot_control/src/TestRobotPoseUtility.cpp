#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "robot_control/RobotPoseUtility.hpp"
#include <thread>
#include <chrono>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<rclcpp::Node>("test_robot_pose_utility");
    
    RCLCPP_INFO(node->get_logger(), "========== Testing RobotPoseUtility ==========");
    
    // Create utility
    auto pose_utility = std::make_shared<RobotPoseUtility>(node);
    
    // Initialize
    if (!pose_utility->initialize()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize RobotPoseUtility!");
        rclcpp::shutdown();
        return 1;
    }
    
    // Set slow speed for safety
    pose_utility->setVelocityScaling(0.1);
    pose_utility->setAccelerationScaling(0.1);
    
    // Test 1: Read current pose
    RCLCPP_INFO(node->get_logger(), "\n========== Test 1: Reading Current Pose ==========");
    auto current_pose = pose_utility->getCurrentPose();
    if (current_pose) {
        RCLCPP_INFO(node->get_logger(), "Current Pose:");
        RCLCPP_INFO(node->get_logger(), "  Position: [%.4f, %.4f, %.4f]",
                   current_pose->position.x, current_pose->position.y, current_pose->position.z);
        RCLCPP_INFO(node->get_logger(), "  Orientation: [%.3f, %.3f, %.3f, %.3f]",
                   current_pose->orientation.x, current_pose->orientation.y,
                   current_pose->orientation.z, current_pose->orientation.w);
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to get current pose!");
    }
    
    // Test 2: Read current joint values
    RCLCPP_INFO(node->get_logger(), "\n========== Test 2: Reading Joint Values ==========");
    auto joint_values = pose_utility->getCurrentJointValues();
    if (!joint_values.empty()) {
        RCLCPP_INFO(node->get_logger(), "Current Joint Values:");
        for (size_t i = 0; i < joint_values.size(); ++i) {
            RCLCPP_INFO(node->get_logger(), "  Joint %zu: %.4f rad (%.2f deg)",
                       i, joint_values[i], joint_values[i] * 180.0 / M_PI);
        }
    }
    
    // Save starting position
    auto start_joint_values = joint_values;
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Test 3: Move to named target
    RCLCPP_INFO(node->get_logger(), "\n========== Test 3: Moving to 'home' ==========");
    if (pose_utility->moveToNamedTarget("home")) {
        RCLCPP_INFO(node->get_logger(), "Successfully moved to 'home'!");
        
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        auto home_pose = pose_utility->getCurrentPose();
        if (home_pose) {
            RCLCPP_INFO(node->get_logger(), "Pose at 'home': [%.4f, %.4f, %.4f]",
                       home_pose->position.x, home_pose->position.y, home_pose->position.z);
        }
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to move to 'home'!");
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Test 4: Move back to starting position
    if (!start_joint_values.empty()) {
        RCLCPP_INFO(node->get_logger(), "\n========== Test 4: Moving Back to Start ==========");
        if (pose_utility->moveToJointValues(start_joint_values)) {
            RCLCPP_INFO(node->get_logger(), "Successfully returned to starting position!");
            
            std::this_thread::sleep_for(std::chrono::seconds(3));
            
            auto final_pose = pose_utility->getCurrentPose();
            if (final_pose) {
                RCLCPP_INFO(node->get_logger(), "Final Pose: [%.4f, %.4f, %.4f]",
                           final_pose->position.x, final_pose->position.y, final_pose->position.z);
            }
        } else {
            RCLCPP_ERROR(node->get_logger(), "Failed to return to starting position!");
        }
    }
    
    RCLCPP_INFO(node->get_logger(), "\n========== All Tests Complete ==========");
    
    rclcpp::shutdown();
    return 0;
}