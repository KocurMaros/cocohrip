// ur_position_control.hpp
#ifndef UR_POSITION_CONTROL_HPP
#define UR_POSITION_CONTROL_HPP

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/string.hpp>

#include <thread>

class URPositionControl : public rclcpp::Node {
public:
    URPositionControl();
    void init();
    void moveRobot();

private:
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr hand_rpy_L_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr hand_rpy_R_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_L_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_R_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr double_gesture_sub_;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

    // Callbacks
    void handRPYLCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void handRPYRCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void gestureLCallback(const std_msgs::msg::String::SharedPtr msg);
    void gestureRCallback(const std_msgs::msg::String::SharedPtr msg);
    void doubleGestureCallback(const std_msgs::msg::String::SharedPtr msg);
    
    //Variables
    geometry_msgs::msg::Vector3 rpy_left_;
    geometry_msgs::msg::Vector3 rpy_right_;
    std_msgs::msg::String gesture_left_;
    std_msgs::msg::String gesture_right_;
    std_msgs::msg::String gesture_double_;
    bool emergency_stop_;
};

#endif // UR_POSITION_CONTROL_HPP

