#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/header.hpp"

#include "leap_gesture_interface/msg/leap_hand.hpp"
#include "leap_gesture_interface/msg/leap_finger.hpp"
#include "leap_gesture_interface/msg/leap_frame.hpp"
#include "leap_gesture_interface/srv/intersect_finite_cylinder.hpp"
#include "leap_gesture_interface/srv/intersect_infinite_cylinder.hpp"
#include "leap_gesture_interface/srv/intersect_disk.hpp"
#include "leap_gesture_interface/intersection_library.hpp"

namespace leap_gesture_interface {

// Enum for finger types (must match LeapFinger.msg values)
enum FingerType {
    THUMB  = 0,
    INDEX  = 1,
    MIDDLE = 2,
    RING   = 3,
    PINKY  = 4
};

// Enum for joint indices in the joints[] array
enum JointType {
    METACARPAL   = 0,
    PROXIMAL     = 1,
    INTERMEDIATE = 2,
    DISTAL       = 3,
    TIP          = 4
};

} // namespace leap_gesture_interface

using leap_gesture_interface::msg::LeapHand;
using leap_gesture_interface::msg::LeapFrame;
using leap_gesture_interface::msg::LeapFinger;
using leap_gesture_interface::FingerType;
using leap_gesture_interface::JointType;
using leap_gesture_interface::srv::IntersectFiniteCylinder;
using leap_gesture_interface::srv::IntersectInfiniteCylinder;
using leap_gesture_interface::srv::IntersectDisk;

class IntersectionNode : public rclcpp::Node {
public:
    IntersectionNode()
    : Node("intersection_node")
    {
        using std::placeholders::_1;

        sub_ = this->create_subscription<LeapFrame>(
            "/leap_frame", 10, std::bind(&IntersectionNode::callback, this, _1));

        left_raw_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/intersection_point/left/raw", 10);
        left_transformed_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/intersection_point/left/transformed", 10);
        right_raw_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/intersection_point/right/raw", 10);
        right_transformed_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/intersection_point/right/transformed", 10);

        finite_cylinder_service_ = this->create_service<IntersectFiniteCylinder>(
            "intersect_finite_cylinder",
            std::bind(&IntersectionNode::handle_finite_cylinder_request, this, std::placeholders::_1, std::placeholders::_2));

        infinite_cylinder_service_ = this->create_service<IntersectInfiniteCylinder>(
            "intersect_infinite_cylinder",
            std::bind(&IntersectionNode::handle_infinite_cylinder_request, this, std::placeholders::_1, std::placeholders::_2));

        disk_service_ = this->create_service<IntersectDisk>(
            "intersect_disk",
            std::bind(&IntersectionNode::handle_disk_request, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "IntersectionNode ready.");
    }

private:
    geometry_msgs::msg::Point to_ros_frame(const geometry_msgs::msg::Point& ultra) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = ultra.x;
        ros_point.y = -ultra.z;
        ros_point.z = ultra.y;
        return ros_point;
    }

    void callback(const LeapFrame::SharedPtr msg)
    {
        for (const auto& hand : msg->hands) {
            const LeapFinger* index_finger = nullptr;
            for (const auto& finger : hand.fingers) {
                if (finger.finger_type == FingerType::INDEX &&
                    finger.joints.size() > JointType::PROXIMAL &&
                    finger.joints.size() > JointType::TIP) {
                    index_finger = &finger;
                    break;
                }
            }

            if (!index_finger) {
                RCLCPP_WARN(this->get_logger(), "Index finger not found or insufficient joints in a hand.");
                continue;
            }

            const auto& tip_raw = index_finger->joints[JointType::TIP];
            const auto& prox_raw = index_finger->joints[JointType::PROXIMAL];

            // Raw intersection with plane y=0 (in Ultraleap coordinates)
            IntersectionLibrary::Vector3 tip_raw_pt(tip_raw.x, tip_raw.y, tip_raw.z);
            IntersectionLibrary::Vector3 prox_raw_pt(prox_raw.x, prox_raw.y, prox_raw.z);
            IntersectionLibrary::Vector3 raw_plane_point(0.0, 0.0, 0.0);
            IntersectionLibrary::Vector3 raw_plane_normal(0.0, 1.0, 0.0);

            auto [intersects_raw, raw_result] = IntersectionLibrary::intersectLinePlane(
                prox_raw_pt, tip_raw_pt, raw_plane_point, raw_plane_normal);

            if (!intersects_raw) {
                RCLCPP_WARN(this->get_logger(), "No intersection with raw plane y=0.");
                continue;
            }

            geometry_msgs::msg::Point raw_point;
            raw_point.x = raw_result.x;
            raw_point.y = raw_result.y;
            raw_point.z = raw_result.z;

            // Transform both points to ROS and compute intersection with Z=0 plane
            auto tip_ros = to_ros_frame(tip_raw);
            auto prox_ros = to_ros_frame(prox_raw);
            IntersectionLibrary::Vector3 tip_pt(tip_ros.x, tip_ros.y, tip_ros.z);
            IntersectionLibrary::Vector3 prox_pt(prox_ros.x, prox_ros.y, prox_ros.z);
            IntersectionLibrary::Vector3 plane_point(0.0, 0.0, 0.0);
            IntersectionLibrary::Vector3 plane_normal(0.0, 0.0, 1.0);

            auto [intersects_transformed, transformed_result] = IntersectionLibrary::intersectLinePlane(
                prox_pt, tip_pt, plane_point, plane_normal);

            if (!intersects_transformed) {
                RCLCPP_WARN(this->get_logger(), "No intersection with transformed Z=0 plane.");
                continue;
            }

            geometry_msgs::msg::Point transformed_point;
            transformed_point.x = transformed_result.x;
            transformed_point.y = transformed_result.y;
            transformed_point.z = transformed_result.z;

            geometry_msgs::msg::PointStamped raw_msg;
            raw_msg.header = msg->header;
            raw_msg.point = raw_point;

            geometry_msgs::msg::PointStamped transformed_msg;
            transformed_msg.header = msg->header;
            transformed_msg.point = transformed_point;

            if (hand.hand_id == 0) {
                left_raw_pub_->publish(raw_msg);
                left_transformed_pub_->publish(transformed_msg);
            } else {
                right_raw_pub_->publish(raw_msg);
                right_transformed_pub_->publish(transformed_msg);
            }
        }
    }

    void handle_finite_cylinder_request(const std::shared_ptr<IntersectFiniteCylinder::Request> request,
                                        std::shared_ptr<IntersectFiniteCylinder::Response> response) {
        auto axis = IntersectionLibrary::Vector3(
            request->cylinder_tip.x - request->cylinder_base.x,
            request->cylinder_tip.y - request->cylinder_base.y,
            request->cylinder_tip.z - request->cylinder_base.z);

        auto result = IntersectionLibrary::intersectLineFiniteCylinder(
            {request->line_point1.x, request->line_point1.y, request->line_point1.z},
            {request->line_point2.x, request->line_point2.y, request->line_point2.z},
            {request->cylinder_base.x, request->cylinder_base.y, request->cylinder_base.z},
            axis,
            request->height, request->radius);

        for (const auto& pt : result) {
            geometry_msgs::msg::Point p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            response->intersections.push_back(p);
        }
    }

    void handle_infinite_cylinder_request(const std::shared_ptr<IntersectInfiniteCylinder::Request> request,
                                          std::shared_ptr<IntersectInfiniteCylinder::Response> response) {
        auto result = IntersectionLibrary::intersectLineCylinder(
            {request->line_point1.x, request->line_point1.y, request->line_point1.z},
            {request->line_point2.x, request->line_point2.y, request->line_point2.z},
            {request->cylinder_axis_start.x, request->cylinder_axis_start.y, request->cylinder_axis_start.z},
            {request->cylinder_axis_end.x, request->cylinder_axis_end.y, request->cylinder_axis_end.z},
            request->radius);

        for (const auto& pt : result) {
            geometry_msgs::msg::Point p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            response->intersections.push_back(p);
        }
    }

    void handle_disk_request(const std::shared_ptr<IntersectDisk::Request> request,
                             std::shared_ptr<IntersectDisk::Response> response) {
        auto [intersects, pt] = IntersectionLibrary::intersectLineDisk(
            {request->line_point1.x, request->line_point1.y, request->line_point1.z},
            {request->line_point2.x, request->line_point2.y, request->line_point2.z},
            {request->disk_center.x, request->disk_center.y, request->disk_center.z},
            {request->disk_normal.x, request->disk_normal.y, request->disk_normal.z},
            request->radius);

        response->intersects = intersects;
        if (intersects) {
            response->intersection.x = pt.x;
            response->intersection.y = pt.y;
            response->intersection.z = pt.z;
        }
    }

    rclcpp::Subscription<LeapFrame>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr left_raw_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr left_transformed_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr right_raw_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr right_transformed_pub_;
    rclcpp::Service<IntersectFiniteCylinder>::SharedPtr finite_cylinder_service_;
    rclcpp::Service<IntersectInfiniteCylinder>::SharedPtr infinite_cylinder_service_;
    rclcpp::Service<IntersectDisk>::SharedPtr disk_service_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IntersectionNode>());
    rclcpp::shutdown();
    return 0;
}
