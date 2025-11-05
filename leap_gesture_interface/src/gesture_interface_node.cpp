#include "rclcpp/rclcpp.hpp"
#include "leap_gesture_interface/msg/leap_frame.hpp"
#include "leap_gesture_interface/msg/recognized_gesture.hpp"
#include "leap_gesture_interface/msg/gesture_event.hpp"
#include "leap_gesture_interface/msg/gesture_pair.hpp"
#include <unordered_set>
#include <string_view>

// Bitmask encoding (finger_type):
// Thumb = 0 -> 00001
// Index = 1 -> 00010
// Middle = 2 -> 00100
// Ring = 3 -> 01000
// Pinky = 4 -> 10000
// Full hand: Thumb+Index+Middle+Ring+Pinky = 11111

using std::placeholders::_1;
using namespace leap_gesture_interface::msg;

class GestureInterfaceNode : public rclcpp::Node
{
public:
    enum class Gesture : uint8_t {
        FIST, THUMB, POINT, PINKY, PISTOL, VICTORY,
        HORNS, SHAKA, THREE, ROCKER, FOUR, FIVE, UNKNOWN
    };

    GestureInterfaceNode() : Node("gesture_interface_node")
    {
        filter_frames_ = this->declare_parameter<int>("filter_frames", 5);
        init_gesture_ = stringToGesture(this->declare_parameter<std::string>("init_gesture", "FIST"));

        pub_recognized_ = create_publisher<RecognizedGesture>("/leap_gesture/recognized", 10);
        pub_event_ = create_publisher<GestureEvent>("/leap_gesture/event", 10);
        pub_pair_ = create_publisher<GesturePair>("/leap_gesture/pair", 10);

        frame_sub_ = create_subscription<LeapFrame>(
    "/leap_frame",
    rclcpp::SensorDataQoS(),
    std::bind(&GestureInterfaceNode::frameCallback, this, _1)
);
    }

private:
    rclcpp::Subscription<LeapFrame>::SharedPtr frame_sub_;
    rclcpp::Publisher<RecognizedGesture>::SharedPtr pub_recognized_;
    rclcpp::Publisher<GestureEvent>::SharedPtr pub_event_;
    rclcpp::Publisher<GesturePair>::SharedPtr pub_pair_;

    Gesture current_L_ = Gesture::UNKNOWN, current_R_ = Gesture::UNKNOWN;
    Gesture previous_L_ = Gesture::UNKNOWN, previous_R_ = Gesture::UNKNOWN;
    Gesture init_gesture_ = Gesture::FIST;

    uint8_t counter_L_ = 0, init_counter_L_ = 0;
    uint8_t counter_R_ = 0, init_counter_R_ = 0;
    int filter_frames_ = 5;

    Gesture stringToGesture(const std::string &name)
    {
        static const std::unordered_map<std::string, Gesture> map = {
            {"FIST", Gesture::FIST}, {"THUMB", Gesture::THUMB}, {"POINT", Gesture::POINT}, {"PINKY", Gesture::PINKY},
            {"PISTOL", Gesture::PISTOL}, {"VICTORY", Gesture::VICTORY}, {"HORNS", Gesture::HORNS}, {"SHAKA", Gesture::SHAKA},
            {"THREE", Gesture::THREE}, {"ROCKER", Gesture::ROCKER}, {"FOUR", Gesture::FOUR}, {"FIVE", Gesture::FIVE}};
        auto it = map.find(name);
        return it != map.end() ? it->second : Gesture::UNKNOWN;
    }

    static std::string_view gestureToStr(Gesture g)
    {
        static constexpr std::string_view names[] = {
            "FIST", "THUMB", "POINT", "PINKY", "PISTOL", "VICTORY",
            "HORNS", "SHAKA", "THREE", "ROCKER", "FOUR", "FIVE", "UNKNOWN"
        };
        return names[static_cast<uint8_t>(g)];
    }

    Gesture recognizeGesture(const std::vector<LeapFinger> &fingers)
    {
        uint8_t mask = 0;
        for (const auto &f : fingers)
            if (f.is_extended) mask |= (1 << f.finger_type);

        switch (__builtin_popcount(mask)) {
            case 0: return Gesture::FIST;
            case 1:
                if (mask & (1 << 0)) return Gesture::THUMB;
                if (mask & (1 << 1)) return Gesture::POINT;
                if (mask & (1 << 4)) return Gesture::PINKY;
                break;
            case 2:
                if ((mask & ((1 << 0) | (1 << 1))) == ((1 << 0) | (1 << 1))) return Gesture::PISTOL;
                if ((mask & ((1 << 1) | (1 << 2))) == ((1 << 1) | (1 << 2))) return Gesture::VICTORY;
                if ((mask & ((1 << 1) | (1 << 4))) == ((1 << 1) | (1 << 4))) return Gesture::HORNS;
                if ((mask & ((1 << 0) | (1 << 4))) == ((1 << 0) | (1 << 4))) return Gesture::SHAKA;
                break;
            case 3:
                if ((mask & ((1 << 1) | (1 << 2))) && ((mask & ((1 << 0) | (1 << 3))) != 0)) return Gesture::THREE;
                if ((mask & ((1 << 0) | (1 << 1) | (1 << 4))) == ((1 << 0) | (1 << 1) | (1 << 4))) return Gesture::ROCKER;
                break;
            case 4: return Gesture::FOUR;
            case 5: return Gesture::FIVE;
        }
        return Gesture::UNKNOWN;
    }

    bool updateCounter(Gesture current, Gesture previous, uint8_t &counter)
    {
        if (current == previous)
            return ++counter == filter_frames_;
        counter = 0;
        return false;
    }

    bool checkInit(Gesture current, Gesture previous, uint8_t &counter)
    {
        if (current == init_gesture_) {
            if (counter < filter_frames_) ++counter;
        } else if (current != previous && counter == filter_frames_) {
            counter = 0;
            return true;
        }
        return false;
    }

    void publishEvent(const std::string &type, const std::string_view &name, bool is_left, const std_msgs::msg::Header &header)
    {
        GestureEvent msg;
        msg.header = header;
        msg.event_type = type;
        msg.gesture_name = std::string(name);
        msg.is_left = is_left;
        pub_event_->publish(msg);
    }

    void frameCallback(const LeapFrame::SharedPtr msg)
    {
        for (const auto &hand : msg->hands)
        {
            const auto gesture = recognizeGesture(hand.fingers);
            const auto name = gestureToStr(gesture);

            RecognizedGesture rg;
            rg.header = msg->header;
            rg.is_left = hand.is_left;
            rg.gesture_name = std::string(name);
            pub_recognized_->publish(rg);

            if (hand.is_left) {
                current_L_ = gesture;
                if (updateCounter(gesture, previous_L_, counter_L_))
                    publishEvent("filtered", name, true, msg->header);
                if (checkInit(gesture, previous_L_, init_counter_L_))
                    publishEvent("initiated", name, true, msg->header);
                previous_L_ = gesture;
            } else {
                current_R_ = gesture;
                if (updateCounter(gesture, previous_R_, counter_R_))
                    publishEvent("filtered", name, false, msg->header);
                if (checkInit(gesture, previous_R_, init_counter_R_))
                    publishEvent("initiated", name, false, msg->header);
                previous_R_ = gesture;
            }
        }

        if (current_L_ == current_R_ && current_L_ != Gesture::UNKNOWN) {
            GesturePair gp;
            gp.header = msg->header;
            gp.left_gesture = std::string(gestureToStr(current_L_));
            gp.right_gesture = std::string(gestureToStr(current_R_));
            gp.match = true;
            pub_pair_->publish(gp);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GestureInterfaceNode>());
    rclcpp::shutdown();
    return 0;
}
