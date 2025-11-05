// ur_position_control.cpp
#include "cocohrip_moveit_control/ur_position_control.hpp"

URPositionControl::URPositionControl() : Node("ur_position_control") {

    // Initialize variables
    rpy_left_.x = 0;
    rpy_left_.y = 0;
    rpy_left_.z = 0;
    rpy_right_.x = 0;
    rpy_right_.y = 0;
    rpy_right_.z = 0;
    gesture_left_.data = "UNKNOWN";
    gesture_right_.data = "UNKNOWN";
    gesture_double_.data = "UNKNOWN";
    emergency_stop_ = false;
}

void URPositionControl::init() {
    // Initialize subscribers
    hand_rpy_L_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "leap_gesture/normalized_rpy_left", 10, std::bind(&URPositionControl::handRPYLCallback, this, std::placeholders::_1));
    hand_rpy_R_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "leap_gesture/normalized_rpy_right", 10, std::bind(&URPositionControl::handRPYRCallback, this, std::placeholders::_1));
    gesture_L_sub_ = this->create_subscription<std_msgs::msg::String>(
            "leap_gesture/filtered_left", 10, std::bind(&URPositionControl::gestureLCallback, this, std::placeholders::_1));
    gesture_R_sub_ = this->create_subscription<std_msgs::msg::String>(
            "leap_gesture/filtered_right", 10, std::bind(&URPositionControl::gestureRCallback, this, std::placeholders::_1));
    double_gesture_sub_ = this->create_subscription<std_msgs::msg::String>(
            "leap_gesture/double", 10, std::bind(&URPositionControl::doubleGestureCallback, this, std::placeholders::_1));

    // Initialize MoveIt
    static const std::string PLANNING_GROUP = "ur_manipulator";
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), PLANNING_GROUP);
}

// Implement callbacks here
void URPositionControl::handRPYLCallback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
    rpy_left_ = *msg;
    std::cout << "Bola zachytena ruka" << std::endl;
}

void URPositionControl::handRPYRCallback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
    rpy_right_ = *msg;
    std::cout << "Bola zachytena ruka" << std::endl;
}

void URPositionControl::gestureLCallback(const std_msgs::msg::String::SharedPtr msg) {
    gesture_left_ = *msg;
    std::cout << "Bola zachytena ruka" << std::endl;
}

void URPositionControl::gestureRCallback(const std_msgs::msg::String::SharedPtr msg) {
    gesture_right_ = *msg;
    std::cout << "Bola zachytena ruka" << std::endl;
}

void URPositionControl::doubleGestureCallback(const std_msgs::msg::String::SharedPtr msg) {
    gesture_double_ = *msg;
    if (gesture_double_.data == "DOUBLE FIST") {
        emergency_stop_ = true;
    }
    std::cout << "Bola zachytena ruka" << std::endl;
}

void URPositionControl::moveRobot() {
    if (gesture_right_.data == "ROCKER") {
        //Set the position of the target pose for ROCKER
        move_group_->setNamedTarget("wave_1");
        move_group_->move();
        move_group_->setNamedTarget("wave_2");
        move_group_->move();
        move_group_->setNamedTarget("wave_1");
        move_group_->move();
        std::cout << "Bola zachytena ruka" << std::endl;
    } else if (gesture_right_.data == "PINKY") {
        move_group_->setNamedTarget("up");
        move_group_->move();
        std::cout << "Bola zachytena ruka" << std::endl;
    } else if (gesture_right_.data == "SHAKA") {
        move_group_->setNamedTarget("home");
        move_group_->move();
        std::cout << "Bola zachytena ruka" << std::endl;
    } else if (gesture_right_.data == "THUMB") {
        move_group_->setNamedTarget("test_configuration");
        move_group_->move();
        std::cout << "Bola zachytena ruka" << std::endl;
    } else if (gesture_right_.data == "PISTOL") {
        //move_group_->setNamedTarget("handoff");
        //move_group_->move();
        //std::cout << "Bola zachytena ruka" << std::endl;
    } else {
        // For other gestures, you can add more else if conditions here
        // For now, if the gesture is not recognized, do not move the robot
        return;
    }
}


