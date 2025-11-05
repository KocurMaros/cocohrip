#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "robot_control/RobotPoseUtility.hpp"
#include <thread>
#include <chrono>

/**
 * @brief Test Cartesian movements from current position
 * 
 * This test specifically focuses on Cartesian pose movements to diagnose issues
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<rclcpp::Node>("test_cartesian_movement");
    
    RCLCPP_INFO(node->get_logger(), "╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║      Cartesian Movement Diagnostic Test          ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝\n");
    
    // Create and initialize utility
    auto pose_utility = std::make_shared<RobotPoseUtility>(node);
    
    if (!pose_utility->initialize()) {
        RCLCPP_ERROR(node->get_logger(), "❌ Failed to initialize RobotPoseUtility!");
        rclcpp::shutdown();
        return 1;
    }
    
    // Set very slow speed for safety
    pose_utility->setVelocityScaling(0.05);  // 5% speed - very slow
    pose_utility->setAccelerationScaling(0.05);
    RCLCPP_INFO(node->get_logger(), "⚙️  Set speed to 5%% for maximum safety\n");
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // =============================================================
    // TEST 1: Read current position
    // =============================================================
    RCLCPP_INFO(node->get_logger(), "╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║ Test 1: Reading Current Position                 ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
    
    auto current_pose = pose_utility->getCurrentPose();
    if (!current_pose) {
        RCLCPP_ERROR(node->get_logger(), "❌ Failed to get current pose!");
        rclcpp::shutdown();
        return 1;
    }
    
    auto current_joints = pose_utility->getCurrentJointValues();
    
    RCLCPP_INFO(node->get_logger(), "\n📍 Current Robot State:");
    RCLCPP_INFO(node->get_logger(), "  Position: [x: %.4f, y: %.4f, z: %.4f]",
               current_pose->position.x, current_pose->position.y, current_pose->position.z);
    RCLCPP_INFO(node->get_logger(), "  Orientation: [x: %.4f, y: %.4f, z: %.4f, w: %.4f]",
               current_pose->orientation.x, current_pose->orientation.y,
               current_pose->orientation.z, current_pose->orientation.w);
    
    if (!current_joints.empty()) {
        RCLCPP_INFO(node->get_logger(), "\n  Joint Values:");
        const std::vector<std::string> joint_names = {
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3"
        };
        for (size_t i = 0; i < current_joints.size(); ++i) {
            RCLCPP_INFO(node->get_logger(), "    %s: %.4f rad (%.2f°)",
                       joint_names[i].c_str(), current_joints[i], 
                       current_joints[i] * 180.0 / M_PI);
        }
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
    // =============================================================
    // TEST 2: Tiny movement DOWN in Z (1cm)
    // =============================================================
    RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║ Test 2: Tiny Cartesian Movement (1cm DOWN)       ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
    
    geometry_msgs::msg::Pose target_pose1 = *current_pose;
    target_pose1.position.z -= 0.01;  // 1cm DOWN (safest direction)
    
    RCLCPP_INFO(node->get_logger(), "\n🎯 Test 2a: Moving 1cm DOWN in Z");
    RCLCPP_INFO(node->get_logger(), "  Current Z: %.4f m", current_pose->position.z);
    RCLCPP_INFO(node->get_logger(), "  Target Z:  %.4f m", target_pose1.position.z);
    RCLCPP_INFO(node->get_logger(), "  Delta Z:   %.4f m (%.1f mm)", -0.01, -10.0);
    
    bool success1 = pose_utility->moveToPose(target_pose1);
    
    if (success1) {
        RCLCPP_INFO(node->get_logger(), "\n✅ SUCCESS: 1cm DOWN movement completed!");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Verify position
        auto new_pose = pose_utility->getCurrentPose();
        if (new_pose) {
            double actual_delta_z = new_pose->position.z - current_pose->position.z;
            RCLCPP_INFO(node->get_logger(), "📏 Verification:");
            RCLCPP_INFO(node->get_logger(), "  Expected delta Z: -0.010 m");
            RCLCPP_INFO(node->get_logger(), "  Actual delta Z:   %.4f m", actual_delta_z);
            RCLCPP_INFO(node->get_logger(), "  Error: %.4f mm", 
                       std::abs(actual_delta_z + 0.01) * 1000);
        }
        
        // =============================================================
        // TEST 3: Return to original position
        // =============================================================
        RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
        RCLCPP_INFO(node->get_logger(), "║ Test 3: Return to Original Position              ║");
        RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        RCLCPP_INFO(node->get_logger(), "\n🎯 Moving back up 1cm to original Z");
        bool success2 = pose_utility->moveToPose(*current_pose);
        
        if (success2) {
            RCLCPP_INFO(node->get_logger(), "\n✅ SUCCESS: Returned to original position!");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            // Final verification
            auto final_pose = pose_utility->getCurrentPose();
            if (final_pose) {
                double total_error = std::sqrt(
                    std::pow(final_pose->position.x - current_pose->position.x, 2) +
                    std::pow(final_pose->position.y - current_pose->position.y, 2) +
                    std::pow(final_pose->position.z - current_pose->position.z, 2)
                );
                RCLCPP_INFO(node->get_logger(), "📏 Final Position Error: %.4f mm", 
                           total_error * 1000);
                
                if (total_error < 0.005) {  // 5mm tolerance
                    RCLCPP_INFO(node->get_logger(), "✅ Position error is acceptable (< 5mm)");
                } else {
                    RCLCPP_WARN(node->get_logger(), "⚠️  Position error is larger than expected");
                }
            }
            
            // =============================================================
            // TEST 4: Try movement in X direction
            // =============================================================
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
            RCLCPP_INFO(node->get_logger(), "║ Test 4: Movement in X Direction (1cm)            ║");
            RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
            
            geometry_msgs::msg::Pose target_pose_x = *current_pose;
            target_pose_x.position.x += 0.01;  // 1cm forward
            
            RCLCPP_INFO(node->get_logger(), "\n🎯 Moving 1cm forward in X");
            bool success_x = pose_utility->moveToPose(target_pose_x);
            
            if (success_x) {
                RCLCPP_INFO(node->get_logger(), "\n✅ SUCCESS: X movement completed!");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                // Return
                RCLCPP_INFO(node->get_logger(), "🎯 Returning to original X");
                pose_utility->moveToPose(*current_pose);
                
                RCLCPP_INFO(node->get_logger(), "\n🎉 ALL CARTESIAN TESTS PASSED!");
            } else {
                RCLCPP_WARN(node->get_logger(), "\n⚠️  X movement failed, but Z movements worked");
            }
            
        } else {
            RCLCPP_ERROR(node->get_logger(), "\n❌ FAILED: Could not return to original position!");
        }
        
    } else {
        RCLCPP_ERROR(node->get_logger(), "\n❌ FAILED: 1cm DOWN movement failed!");
        
        // =============================================================
        // FALLBACK TEST: Try even smaller movement (5mm)
        // =============================================================
        RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
        RCLCPP_INFO(node->get_logger(), "║ Fallback Test: Tiny Movement (5mm DOWN)          ║");
        RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
        
        geometry_msgs::msg::Pose target_pose_tiny = *current_pose;
        target_pose_tiny.position.z -= 0.005;  // 5mm DOWN
        
        RCLCPP_INFO(node->get_logger(), "\n🎯 Trying 5mm DOWN movement");
        bool success_tiny = pose_utility->moveToPose(target_pose_tiny);
        
        if (success_tiny) {
            RCLCPP_INFO(node->get_logger(), "\n✅ SUCCESS: 5mm movement worked!");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            pose_utility->moveToPose(*current_pose);
        } else {
            RCLCPP_ERROR(node->get_logger(), "\n❌ Even 5mm movement failed!");
            RCLCPP_ERROR(node->get_logger(), "\n⚠️  DIAGNOSIS:");
            RCLCPP_ERROR(node->get_logger(), "  - Robot may be at workspace limit");
            RCLCPP_ERROR(node->get_logger(), "  - Robot may be near singularity");
            RCLCPP_ERROR(node->get_logger(), "  - Orientation constraints may be too strict");
            RCLCPP_ERROR(node->get_logger(), "\n💡 RECOMMENDATION: Use joint-space movements instead");
        }
    }
    
    // =============================================================
    // SUMMARY
    // =============================================================
    RCLCPP_INFO(node->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(node->get_logger(), "║            Test Sequence Complete                 ║");
    RCLCPP_INFO(node->get_logger(), "╚════════════════════════════════════════════════════╝");
    
    RCLCPP_INFO(node->get_logger(), "\nPress Ctrl+C to exit...");
    
    rclcpp::shutdown();
    return 0;
}