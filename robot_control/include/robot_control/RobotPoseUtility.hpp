#ifndef ROBOT_POSE_UTILITY_HPP
#define ROBOT_POSE_UTILITY_HPP

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <mutex>
#include <string>
#include <vector>

/**
 * @brief Utility class for reading and setting robot poses via MoveIt
 * 
 * This class provides methods to:
 * - Get current end-effector pose
 * - Get current joint values
 * - Move to named targets
 * - Move to specific joint configurations
 * - Move to Cartesian poses
 */
class RobotPoseUtility
{
public:
    /**
     * @brief Constructor
     * @param node Shared pointer to the ROS2 node
     * @param planning_group Name of the MoveIt planning group (default: "ur_manipulator")
     * @param end_effector_link Name of the end effector link (default: "tool0")
     */
    RobotPoseUtility(
        std::shared_ptr<rclcpp::Node> node,
        const std::string& planning_group = "ur5e_arm",
        const std::string& end_effector_link = "ur5e_tool0"
    );

    /**
     * @brief Initialize the utility (loads robot model, creates MoveGroup)
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Get the current pose of the end effector
     * @return Current pose, or empty optional if failed
     */
    std::optional<geometry_msgs::msg::Pose> getCurrentPose();

    /**
     * @brief Get current joint values
     * @return Vector of joint values in order, or empty vector if failed
     */
    std::vector<double> getCurrentJointValues();

    /**
     * @brief Move to a named target configuration
     * @param target_name Name of the target (e.g., "home")
     * @return true if movement successful, false otherwise
     */
    bool moveToNamedTarget(const std::string& target_name);

    /**
     * @brief Move to specific joint values
     * @param joint_values Vector of 6 joint values (in radians)
     * @return true if movement successful, false otherwise
     */
    bool moveToJointValues(const std::vector<double>& joint_values);

    /**
     * @brief Move to a Cartesian pose
     * @param target_pose Target pose for end effector
     * @return true if movement successful, false otherwise
     */
    bool moveToPose(const geometry_msgs::msg::Pose& target_pose);

    /**
     * @brief Set velocity scaling factor
     * @param factor Scaling factor (0.0 to 1.0)
     */
    void setVelocityScaling(double factor);

    /**
     * @brief Set acceleration scaling factor
     * @param factor Scaling factor (0.0 to 1.0)
     */
    void setAccelerationScaling(double factor);

    /**
     * @brief Stop current movement immediately
     * This will halt the robot at its current position
     */
    void stopMovement();

    /**
     * @brief Check if stop was requested
     */
    bool isStopRequested() const { return stop_requested_.load(); }
    
    /**
     * @brief Clear the stop request flag (call before starting new movement)
     */
    void clearStopRequest() { stop_requested_.store(false); }

    /**
     * @brief Check if the utility is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    bool updateRobotState();

    std::shared_ptr<rclcpp::Node> node_;
    std::string planning_group_;
    std::string end_effector_link_;
    
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
    moveit::core::RobotModelPtr robot_model_;
    moveit::core::RobotStatePtr robot_state_;
    
    std::atomic<bool> stop_requested_{false};  // Flag to signal stop request
    
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;
    std::mutex joint_state_mutex_;
    bool joint_state_received_;
    bool initialized_;
    double min_z_offset_;  // Minimum Z height for end effector safety
};

#endif // ROBOT_POSE_UTILITY_HPP