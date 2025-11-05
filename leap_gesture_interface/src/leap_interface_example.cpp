#include "rclcpp/rclcpp.hpp"

extern "C"{
#include "LeapC.h"
#include "ExampleConnection.h"
}

class GestureInterfaceNode : public rclcpp::Node
{
public:
    GestureInterfaceNode() : Node("gesture_interface_node")
    {
        // Initialize LeapC connection
        OpenConnection();
        while (!IsConnected)
        {
            rclcpp::sleep_for(std::chrono::milliseconds(100)); // Wait for the connection to complete
        }

        RCLCPP_INFO(this->get_logger(), "Connected.");
        LEAP_DEVICE_INFO* deviceProps = GetDeviceProperties();
        if (deviceProps)
        {
            RCLCPP_INFO(this->get_logger(), "Using device %s.", deviceProps->serial);
        }
    }

    void run()
    {
        rclcpp::Rate rate(10); // For example, 10Hz rate
        int64_t lastFrameID = 0;

        while (rclcpp::ok())
        {
            LEAP_TRACKING_EVENT *frame = GetFrame();
            if (frame && (frame->tracking_frame_id > lastFrameID))
            {
                lastFrameID = frame->tracking_frame_id;
                RCLCPP_INFO(this->get_logger(), "Frame %lli with %i hands.", (long long int)frame->tracking_frame_id, frame->nHands);

                for (uint32_t h = 0; h < frame->nHands; h++)
                {
                    LEAP_HAND* hand = &frame->pHands[h];
                    RCLCPP_INFO(this->get_logger(), "Hand id %i is a %s hand with position (%f, %f, %f).",
                                hand->id,
                                (hand->type == eLeapHandType_Left ? "left" : "right"),
                                hand->palm.position.x,
                                hand->palm.position.y,
                                hand->palm.position.z);
                }
            }
            rate.sleep();
        }
    }

private:
    // ... (any private member variables or methods you may need)
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GestureInterfaceNode>();
    node->run();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}