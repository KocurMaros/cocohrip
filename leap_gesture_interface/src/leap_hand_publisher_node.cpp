#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include "leap_gesture_interface/msg/leap_hand.hpp"
#include "leap_gesture_interface/msg/leap_finger.hpp"
#include "leap_gesture_interface/msg/leap_palm.hpp"
#include "leap_gesture_interface/msg/leap_frame.hpp"
#include <vector>
#include <cmath>
#include <thread>

extern "C" {
#include "LeapC.h"
#include "ExampleConnection.h"
}

class LeapPublisherNode : public rclcpp::Node {
public:
    LeapPublisherNode() : Node("leap_publisher_node") {
        double publish_rate = this->declare_parameter<double>("publish_rate", 10.0);
        publisher_ = this->create_publisher<leap_gesture_interface::msg::LeapFrame>("/leap_frame", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
            std::bind(&LeapPublisherNode::publish_hand_data, this));

        OpenConnection();
        while (!IsConnected) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO(this->get_logger(), "Connected to Leap Motion.");

        if (const auto *deviceProps = GetDeviceProperties()) {
            RCLCPP_INFO(this->get_logger(), "Using device %s.", deviceProps->serial);
        }
    }

private:
    void publish_hand_data() {
        if (const LEAP_TRACKING_EVENT *frame = GetFrame()) {
            constexpr double mm_to_m = 0.001;
            leap_gesture_interface::msg::LeapFrame frame_msg;
            frame_msg.header.stamp = this->now();
            frame_msg.header.frame_id = "leap_hand_frame";
            frame_msg.n_hands = frame->nHands;

            frame_msg.hands.reserve(frame->nHands);

            for (uint32_t h = 0; h < frame->nHands; ++h) {
                const LEAP_HAND &hand = frame->pHands[h];
                leap_gesture_interface::msg::LeapHand msg;
                msg.header = frame_msg.header;
                msg.hand_id = hand.id;
                msg.is_left = (hand.type == eLeapHandType_Left);

                const auto &p = hand.palm;
                leap_gesture_interface::msg::LeapPalm palm;
                palm.position.x = p.position.v[0] * mm_to_m;
                palm.position.y = p.position.v[1] * mm_to_m;
                palm.position.z = p.position.v[2] * mm_to_m;
                palm.stabilized_position.x = p.stabilized_position.v[0] * mm_to_m;
                palm.stabilized_position.y = p.stabilized_position.v[1] * mm_to_m;
                palm.stabilized_position.z = p.stabilized_position.v[2] * mm_to_m;
                palm.velocity.x = p.velocity.v[0] * mm_to_m;
                palm.velocity.y = p.velocity.v[1] * mm_to_m;
                palm.velocity.z = p.velocity.v[2] * mm_to_m;
                palm.normal.x = p.normal.v[0]; palm.normal.y = p.normal.v[1]; palm.normal.z = p.normal.v[2];
                palm.direction.x = p.direction.v[0]; palm.direction.y = p.direction.v[1]; palm.direction.z = p.direction.v[2];
                palm.orientation.x = p.orientation.x;
                palm.orientation.y = p.orientation.y;
                palm.orientation.z = p.orientation.z;
                palm.orientation.w = p.orientation.w;
                palm.width = p.width * mm_to_m;
                msg.palm = std::move(palm);

                msg.arm_position.x = hand.arm.next_joint.v[0] * mm_to_m;
                msg.arm_position.y = hand.arm.next_joint.v[1] * mm_to_m;
                msg.arm_position.z = hand.arm.next_joint.v[2] * mm_to_m;

                msg.pinch_distance = hand.pinch_distance * mm_to_m;
                msg.grab_angle = hand.grab_angle;
                msg.pinch_strength = hand.pinch_strength;
                msg.grab_strength = hand.grab_strength;

                msg.fingers.reserve(5);
                for (int i = 0; i < 5; ++i) {
                    leap_gesture_interface::msg::LeapFinger finger_msg;
                    finger_msg.finger_type = i;
                    finger_msg.is_extended = static_cast<uint32_t>(hand.digits[i].is_extended);

                    const LEAP_DIGIT &digit = hand.digits[i];
                    finger_msg.joints.reserve(6);

                    geometry_msgs::msg::Point last_joint{};
                    for (int b = 0; b < 4; ++b) {
                        const auto &bone = digit.bones[b];
                        geometry_msgs::msg::Point prev, next;
                        prev.x = bone.prev_joint.v[0] * mm_to_m;
                        prev.y = bone.prev_joint.v[1] * mm_to_m;
                        prev.z = bone.prev_joint.v[2] * mm_to_m;
                        next.x = bone.next_joint.v[0] * mm_to_m;
                        next.y = bone.next_joint.v[1] * mm_to_m;
                        next.z = bone.next_joint.v[2] * mm_to_m;

                        if (b == 0 ||
                            std::abs(prev.x - last_joint.x) > 1e-4 ||
                            std::abs(prev.y - last_joint.y) > 1e-4 ||
                            std::abs(prev.z - last_joint.z) > 1e-4) {
                            finger_msg.joints.push_back(prev);
                        }
                        finger_msg.joints.push_back(next);
                        last_joint = next;
                    }
                    msg.fingers.push_back(std::move(finger_msg));
                }
                frame_msg.hands.push_back(std::move(msg));
            }

            publisher_->publish(frame_msg);
        }
    }

    rclcpp::Publisher<leap_gesture_interface::msg::LeapFrame>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LeapPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    CloseConnection();
    DestroyConnection();
    return 0;
}
