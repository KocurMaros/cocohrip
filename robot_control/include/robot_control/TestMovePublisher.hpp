#ifndef TEST_MOVE_PUBLISHER_HPP
#define TEST_MOVE_PUBLISHER_HPP

#include <rclcpp/rclcpp.hpp>
#include <checkers_msgs/msg/move.hpp>
#include <checkers_msgs/msg/piece.hpp>
#include <checkers_msgs/msg/board.hpp>

class TestMovePublisher : public rclcpp::Node
{
public:
    TestMovePublisher();
    void publishTestBoard();
    void publishSimpleMove();
    void publishMoveWithCapture();
    void publishMultipleCaptures();
    void publishMoveToBoardCenter();
    void publishCornerSquaresTest();  // NEW: Test 4 corners

private:
    rclcpp::Publisher<checkers_msgs::msg::Board>::SharedPtr board_pub_;
    rclcpp::Publisher<checkers_msgs::msg::Move>::SharedPtr move_pub_;
};

#endif // TEST_MOVE_PUBLISHER_HPP