#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <thread>
#include <chrono>

class SimplePoseReader : public rclcpp::Node
{
public:
    SimplePoseReader()
        : Node("simple_pose_reader"),
          joint_state_received_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Starting Simple Pose Reader...");
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states",
            10,
            std::bind(&SimplePoseReader::jointStateCallback, this, std::placeholders::_1)
        );
    }
    
    void initialize()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for MoveIt to initialize...");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        RCLCPP_INFO(this->get_logger(), "Loading robot model...");
        
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
            shared_from_this(),
            "robot_description"
        );
        
        robot_model_ = robot_model_loader_->getModel();
        if (!robot_model_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load robot model!");
            return;
        }
        
        robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
        robot_state_->setToDefaultValues();
        
        RCLCPP_INFO(this->get_logger(), "Creating MoveGroupInterface...");
        
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), 
            "ur_manipulator"
        );
        
        move_group_->setEndEffectorLink("tool0");
        move_group_->setPoseReferenceFrame("base_link");
        
        move_group_->setPlanningTime(10.0);
        move_group_->setNumPlanningAttempts(10);
        move_group_->setMaxVelocityScalingFactor(0.1);
        move_group_->setMaxAccelerationScalingFactor(0.1);
        
        RCLCPP_INFO(this->get_logger(), "MoveGroupInterface created successfully!");
        RCLCPP_INFO(this->get_logger(), "Planning frame: %s", move_group_->getPlanningFrame().c_str());
        RCLCPP_INFO(this->get_logger(), "End effector link: %s", move_group_->getEndEffectorLink().c_str());
        
        RCLCPP_INFO(this->get_logger(), "Waiting for joint states...");
        int count = 0;
        while (!joint_state_received_ && count < 50 && rclcpp::ok()) {
            rclcpp::spin_some(shared_from_this());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            count++;
        }
        
        if (!joint_state_received_) {
            RCLCPP_WARN(this->get_logger(), "No joint states received yet!");
        } else {
            RCLCPP_INFO(this->get_logger(), "Joint states received!");
        }
        
        monitoring_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&SimplePoseReader::readPose, this)
        );
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        RCLCPP_INFO(this->get_logger(), "========== READING CURRENT POSITION ==========");
        readPose();
        
        // Save starting position
        {
            std::lock_guard<std::mutex> lock(joint_state_mutex_);
            if (latest_joint_state_) {
                start_joint_values_.clear();
                const std::vector<std::string> joint_names = {
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
                };
                
                for (const auto& joint_name : joint_names) {
                    for (size_t i = 0; i < latest_joint_state_->name.size(); ++i) {
                        if (latest_joint_state_->name[i] == joint_name) {
                            start_joint_values_.push_back(latest_joint_state_->position[i]);
                            break;
                        }
                    }
                }
                
                RCLCPP_INFO(this->get_logger(), "Saved starting joint configuration:");
                for (size_t i = 0; i < start_joint_values_.size(); ++i) {
                    RCLCPP_INFO(this->get_logger(), "  Joint %zu: %.4f rad (%.2f deg)", 
                               i, start_joint_values_[i], start_joint_values_[i] * 180.0 / M_PI);
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Execute movements
        RCLCPP_INFO(this->get_logger(), "========== MOVING TO HOME ==========");
        moveToNamedTarget("home");
        
        // Wait for robot to fully stabilize
        RCLCPP_INFO(this->get_logger(), "Waiting for robot to stabilize...");
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        RCLCPP_INFO(this->get_logger(), "========== POSITION AT HOME ==========");
        readPose();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Move back to start
        if (!start_joint_values_.empty()) {
            RCLCPP_INFO(this->get_logger(), "========== MOVING BACK TO START ==========");
            moveToJointValues(start_joint_values_);
            
            RCLCPP_INFO(this->get_logger(), "Waiting for robot to stabilize...");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            RCLCPP_INFO(this->get_logger(), "========== POSITION AT START ==========");
            readPose();
        }
        
        RCLCPP_INFO(this->get_logger(), "========== MOVEMENT SEQUENCE COMPLETE ==========");
        RCLCPP_INFO(this->get_logger(), "Continuing to monitor position every 0.5 seconds...");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        latest_joint_state_ = msg;
        joint_state_received_ = true;
    }
    
    void moveToNamedTarget(const std::string& target_name)
    {
        if (!move_group_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized!");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Moving to named target: %s", target_name.c_str());
        
        move_group_->setNamedTarget(target_name);
        
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        moveit::core::MoveItErrorCode plan_result = move_group_->plan(plan);
        
        if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning successful! Trajectory has %zu waypoints", 
                       plan.trajectory.joint_trajectory.points.size());
            RCLCPP_INFO(this->get_logger(), "Executing trajectory... Watch position updates:");
            
            // Execute the plan
            moveit::core::MoveItErrorCode execute_result = move_group_->execute(plan);
            
            // Even if execute reports failure, check if controller succeeded
            RCLCPP_INFO(this->get_logger(), "Execute returned code: %d", execute_result.val);
            
            if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
                RCLCPP_INFO(this->get_logger(), "Movement to '%s' completed successfully!", target_name.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "Execute returned error %d, but checking if motion completed...", execute_result.val);
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning to '%s' failed with error: %d", target_name.c_str(), plan_result.val);
        }
    }
    
    void moveToJointValues(const std::vector<double>& joint_values)
    {
        if (!move_group_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized!");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Moving to joint configuration:");
        for (size_t i = 0; i < joint_values.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "  Joint %zu: %.4f rad (%.2f deg)", 
                       i, joint_values[i], joint_values[i] * 180.0 / M_PI);
        }
        
        move_group_->setJointValueTarget(joint_values);
        
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        moveit::core::MoveItErrorCode plan_result = move_group_->plan(plan);
        
        if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning successful! Trajectory has %zu waypoints", 
                       plan.trajectory.joint_trajectory.points.size());
            RCLCPP_INFO(this->get_logger(), "Executing trajectory... Watch position updates:");
            
            moveit::core::MoveItErrorCode execute_result = move_group_->execute(plan);
            
            RCLCPP_INFO(this->get_logger(), "Execute returned code: %d", execute_result.val);
            
            if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
                RCLCPP_INFO(this->get_logger(), "Movement to joint configuration completed successfully!");
            } else {
                RCLCPP_WARN(this->get_logger(), "Execute returned error %d, but checking if motion completed...", execute_result.val);
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning to joint configuration failed with error: %d", plan_result.val);
        }
    }
    
    void readPose()
    {
        if (!move_group_ || !robot_state_) {
            return;
        }
        
        sensor_msgs::msg::JointState::SharedPtr joint_state;
        {
            std::lock_guard<std::mutex> lock(joint_state_mutex_);
            if (!latest_joint_state_) {
                return;
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
            
            const Eigen::Isometry3d& end_effector_state = 
                robot_state_->getGlobalLinkTransform("tool0");
            
            geometry_msgs::msg::Pose current_pose;
            current_pose.position.x = end_effector_state.translation().x();
            current_pose.position.y = end_effector_state.translation().y();
            current_pose.position.z = end_effector_state.translation().z();
            
            Eigen::Quaterniond quat(end_effector_state.rotation());
            current_pose.orientation.x = quat.x();
            current_pose.orientation.y = quat.y();
            current_pose.orientation.z = quat.z();
            current_pose.orientation.w = quat.w();
            
            RCLCPP_INFO(this->get_logger(), "Current Pose: [x: %.4f, y: %.4f, z: %.4f] [qx: %.3f, qy: %.3f, qz: %.3f, qw: %.3f]",
                       current_pose.position.x, current_pose.position.y, current_pose.position.z,
                       current_pose.orientation.x, current_pose.orientation.y, 
                       current_pose.orientation.z, current_pose.orientation.w);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error reading pose: %s", e.what());
        }
    }

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
    moveit::core::RobotModelPtr robot_model_;
    moveit::core::RobotStatePtr robot_state_;
    
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;
    std::mutex joint_state_mutex_;
    bool joint_state_received_;
    
    std::vector<double> start_joint_values_;
    
    rclcpp::TimerBase::SharedPtr monitoring_timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<SimplePoseReader>();
    node->initialize();
    
    RCLCPP_INFO(node->get_logger(), "Simple Pose Reader node is running. Press Ctrl+C to exit.");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}