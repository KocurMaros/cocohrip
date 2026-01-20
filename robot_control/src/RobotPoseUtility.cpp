#include "robot_control/RobotPoseUtility.hpp"
#include <thread>
#include <chrono>

RobotPoseUtility::RobotPoseUtility(
    std::shared_ptr<rclcpp::Node> node,
    const std::string& planning_group,
    const std::string& end_effector_link)
    : node_(node),
      planning_group_(planning_group),
      end_effector_link_(end_effector_link),
      joint_state_received_(false),
      initialized_(false),
      min_z_offset_(0.16)  // End effector safety offset

{
    // Subscribe to joint states
    joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states",
        10,
        std::bind(&RobotPoseUtility::jointStateCallback, this, std::placeholders::_1)
    );
}

bool RobotPoseUtility::initialize()
{
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Initializing...");
    
    // Wait a bit for ROS to stabilize
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Load robot model
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Loading robot model...");
    robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
        node_,
        "robot_description"
    );
    
    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Failed to load robot model!");
        return false;
    }
    
    robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();
    
    // Create MoveGroup interface
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Creating MoveGroupInterface...");
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        node_, 
        planning_group_
    );
    
    // Configure MoveGroup with better parameters
    move_group_->setEndEffectorLink(end_effector_link_);
    move_group_->setPoseReferenceFrame("base_link");
    
    // CRITICAL: Set better planning parameters
    move_group_->setPlanningTime(15.0);  // Increased from 10.0
    move_group_->setNumPlanningAttempts(20);  // Increased from 10
    move_group_->setGoalPositionTolerance(0.001);  // 1mm tolerance
    move_group_->setGoalOrientationTolerance(0.01);  // ~0.5 degrees
    
    // Set planner to OMPL's RRTConnect (good for manipulation)
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    
    // Enable replanning
    move_group_->allowReplanning(true);
    
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] MoveGroup configuration:");
    RCLCPP_INFO(node_->get_logger(), "  - Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(node_->get_logger(), "  - End effector: %s", end_effector_link_.c_str());
    RCLCPP_INFO(node_->get_logger(), "  - Planning time: %.1f s", 15.0);
    RCLCPP_INFO(node_->get_logger(), "  - Planning attempts: %d", 20);
    
    // Wait for joint states
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Waiting for joint states...");
    int count = 0;
    while (!joint_state_received_ && count < 50 && rclcpp::ok()) {
        rclcpp::spin_some(node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        count++;
    }
    
    if (!joint_state_received_) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] No joint states received!");
        return false;
    }
    
    initialized_ = true;
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Initialization complete!");
    return true;
}

void RobotPoseUtility::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    latest_joint_state_ = msg;
    joint_state_received_ = true;
}

bool RobotPoseUtility::updateRobotState()
{
    if (!initialized_ || !robot_state_) {
        return false;
    }
    
    sensor_msgs::msg::JointState::SharedPtr joint_state;
    {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        if (!latest_joint_state_) {
            return false;
        }
        joint_state = latest_joint_state_;
    }
    
    try {
        for (size_t i = 0; i < joint_state->name.size(); ++i) {
            if (robot_model_->hasJointModel(joint_state->name[i])) {
                robot_state_->setJointPositions(joint_state->name[i], &joint_state->position[i]);
            }
        }
        robot_state_->update();
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Error updating robot state: %s", e.what());
        return false;
    }
}

std::optional<geometry_msgs::msg::Pose> RobotPoseUtility::getCurrentPose()
{
    if (!updateRobotState()) {
        return std::nullopt;
    }
    
    try {
        const Eigen::Isometry3d& end_effector_state = 
            robot_state_->getGlobalLinkTransform(end_effector_link_);
        
        geometry_msgs::msg::Pose pose;
        pose.position.x = end_effector_state.translation().x();
        pose.position.y = end_effector_state.translation().y();
        pose.position.z = end_effector_state.translation().z();
        
        Eigen::Quaterniond quat(end_effector_state.rotation());
        pose.orientation.x = quat.x();
        pose.orientation.y = quat.y();
        pose.orientation.z = quat.z();
        pose.orientation.w = quat.w();
        
        return pose;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Error getting current pose: %s", e.what());
        return std::nullopt;
    }
}

std::vector<double> RobotPoseUtility::getCurrentJointValues()
{
    std::vector<double> joint_values;
    
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    if (!latest_joint_state_) {
        return joint_values;
    }
    
    const std::vector<std::string> joint_names = {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    };
    
    for (const auto& joint_name : joint_names) {
        for (size_t i = 0; i < latest_joint_state_->name.size(); ++i) {
            if (latest_joint_state_->name[i] == joint_name) {
                joint_values.push_back(latest_joint_state_->position[i]);
                break;
            }
        }
    }
    
    return joint_values;
}

bool RobotPoseUtility::moveToNamedTarget(const std::string& target_name)
{
    if (!initialized_ || !move_group_) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Not initialized!");
        return false;
    }
    
    // NOTE: stop_requested_ is NOT cleared here - caller must clear it before starting a movement sequence
    
    // Check if stop was requested before starting
    if (stop_requested_.load()) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stop requested before moveToNamedTarget - aborting");
        return false;
    }
    
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Moving to named target: %s", target_name.c_str());
    
    move_group_->setNamedTarget(target_name);
    
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::core::MoveItErrorCode plan_result = move_group_->plan(plan);
    
    // Check if stop was requested during planning
    if (stop_requested_.load()) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stop requested during planning - aborting");
        return false;
    }
    
    if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Planning successful!");
        
        moveit::core::MoveItErrorCode execute_result = move_group_->execute(plan);
        
        // Check if stop was requested during execution
        if (stop_requested_.load()) {
            RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Movement was stopped");
            return false;
        }
        
        if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Movement completed!");
            return true;
        } else if (execute_result == moveit::core::MoveItErrorCode::PREEMPTED) {
            RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Movement was preempted (stopped)");
            return false;
        } else {
            RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Execution failed with code: %d", execute_result.val);
            return false;
        }
    } else {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Planning failed with code: %d", plan_result.val);
        return false;
    }
}

bool RobotPoseUtility::moveToJointValues(const std::vector<double>& joint_values)
{
    if (!initialized_ || !move_group_) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Not initialized!");
        return false;
    }
    
    if (joint_values.size() != 6) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Expected 6 joint values, got %zu", joint_values.size());
        return false;
    }
    
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Moving to joint configuration");
    
    move_group_->setJointValueTarget(joint_values);
    
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::core::MoveItErrorCode plan_result = move_group_->plan(plan);
    
    if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
        moveit::core::MoveItErrorCode execute_result = move_group_->execute(plan);
        return (execute_result == moveit::core::MoveItErrorCode::SUCCESS);
    } else {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Joint space planning failed with code: %d", plan_result.val);
    }
    
    return false;
}

bool RobotPoseUtility::moveToPose(const geometry_msgs::msg::Pose& target_pose)
{
    if (!initialized_ || !move_group_) {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Not initialized!");
        return false;
    }
    
    // NOTE: stop_requested_ is NOT cleared here - caller must clear it before starting a movement sequence
    // This allows stop to persist across multiple moveToPose calls in a sequence
    
    geometry_msgs::msg::Pose adjusted_pose = target_pose;
    
    if (adjusted_pose.position.z < min_z_offset_) {
        RCLCPP_WARN(node_->get_logger(), 
                    "[RobotPoseUtility] Target Z (%.4f) below minimum (%.4f)! Clamping to minimum.",
                    adjusted_pose.position.z, min_z_offset_);
        adjusted_pose.position.z = min_z_offset_;
    }
    
    // Check if stop was requested before planning
    if (stop_requested_.load()) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stop requested before planning - aborting");
        return false;
    }
    
    RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Moving to Cartesian pose");
    RCLCPP_DEBUG(node_->get_logger(), "  Target: [%.4f, %.4f, %.4f]",
                target_pose.position.x, target_pose.position.y, target_pose.position.z);
    
    // METHOD 1: Try standard pose target planning (most reliable in Jazzy)
    move_group_->setPoseTarget(adjusted_pose);
    move_group_->setStartStateToCurrentState();
    
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::core::MoveItErrorCode plan_result = move_group_->plan(plan);
    
    // Check if stop was requested during planning
    if (stop_requested_.load()) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stop requested during planning - aborting");
        return false;
    }
    
    if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Planning successful!");
        
        // Check if stop was requested before execution
        if (stop_requested_.load()) {
            RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stop requested before execution - aborting");
            return false;
        }
        
        // Execute the plan - this is blocking but stop() can interrupt it
        moveit::core::MoveItErrorCode execute_result = move_group_->execute(plan);
        
        // Check if stop was requested during execution
        if (stop_requested_.load()) {
            RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Movement was stopped");
            return false;
        }
        
        if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(node_->get_logger(), "[RobotPoseUtility] Movement successful!");
            return true;
        } else if (execute_result == moveit::core::MoveItErrorCode::PREEMPTED) {
            RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Movement was preempted (stopped)");
            return false;
        } else {
            RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Execution failed with code: %d", execute_result.val);
            return false;
        }
    } else {
        RCLCPP_ERROR(node_->get_logger(), "[RobotPoseUtility] Planning failed with code: %d", plan_result.val);
        
        // Try to give more diagnostic information
        auto current_pose = getCurrentPose();
        if (current_pose) {
            double distance = std::sqrt(
                std::pow(target_pose.position.x - current_pose->position.x, 2) +
                std::pow(target_pose.position.y - current_pose->position.y, 2) +
                std::pow(target_pose.position.z - current_pose->position.z, 2)
            );
            RCLCPP_ERROR(node_->get_logger(), "  Distance to target: %.4f m (%.1f mm)", distance, distance * 1000);
            RCLCPP_ERROR(node_->get_logger(), "  Current: [%.4f, %.4f, %.4f]",
                        current_pose->position.x, current_pose->position.y, current_pose->position.z);
            RCLCPP_ERROR(node_->get_logger(), "  Target:  [%.4f, %.4f, %.4f]",
                        target_pose.position.x, target_pose.position.y, target_pose.position.z);
        }
    }
    
    return false;
}

void RobotPoseUtility::setVelocityScaling(double factor)
{
    if (move_group_) {
        move_group_->setMaxVelocityScalingFactor(factor);
        RCLCPP_DEBUG(node_->get_logger(), "[RobotPoseUtility] Velocity scaling set to %.2f", factor);
    }
}

void RobotPoseUtility::setAccelerationScaling(double factor)
{
    if (move_group_) {
        move_group_->setMaxAccelerationScalingFactor(factor);
        RCLCPP_DEBUG(node_->get_logger(), "[RobotPoseUtility] Acceleration scaling set to %.2f", factor);
    }
}

void RobotPoseUtility::stopMovement()
{
    if (move_group_) {
        RCLCPP_WARN(node_->get_logger(), "[RobotPoseUtility] Stopping movement!");
        stop_requested_.store(true);
        move_group_->stop();
    }
}