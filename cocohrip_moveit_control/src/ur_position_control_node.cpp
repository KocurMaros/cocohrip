// ur_position_control_node.cpp
#include "cocohrip_moveit_control/ur_position_control.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<URPositionControl>();
    node->init();

    rclcpp::Rate rate(10); // 10Hz

    while (rclcpp::ok()) {
        node->moveRobot();
        rclcpp::spin_some(node);
        rate.sleep();
    }

    rclcpp::shutdown();


    return 0;
}

