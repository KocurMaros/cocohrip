#ifndef ROBOT_CONTROL_NODE_HPP
#define ROBOT_CONTROL_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include "checkers_msgs/msg/board.hpp" 
#include "checkers_msgs/msg/piece.hpp" 
#include "checkers_msgs/msg/move.hpp" 
#include "checkers_msgs/msg/hand_detected.hpp"
#include "checkers_msgs/msg/robot_move.hpp"
#include "checkers_msgs/srv/resume_movement.hpp"
#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <gripper_srv/srv/gripper_service.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include "robot_control/Mission.hpp"
#include "robot_control/RobotPoseUtility.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include <thread>
#include <geometry_msgs/msg/vector3.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include <mutex>
#include <atomic>
#include <fstream>
#include "leap_gesture_interface/msg/leap_hand.hpp"
#include "leap_gesture_interface/msg/leap_finger.hpp"
#include "leap_gesture_interface/msg/leap_palm.hpp"
#include "leap_gesture_interface/msg/leap_frame.hpp"

class RobotControlNode : public rclcpp::Node
{
public:
    RobotControlNode();
    virtual ~RobotControlNode();

    void publishCheckerboard();
    void mainLoop();
    void initMoveGroup();
    void move(geometry_msgs::msg::Pose targetPose);
    void moveInThread(geometry_msgs::msg::Pose targetPose);
    void stop();
    void continue_move();
    void attachPiece();
    void detachPiece();
    void createPiece(int row, int col);
    void createFakePieceWithColor(const std::string& object_id, int row, int col, const std::string& colorName);
    std::tuple<float, float, float> getColorFromName(const std::string& colorName);
    void removePiece();
    void removeAllFakePieces();
    void removeFakePiece(const std::string& object_id);
    int convertStringToInt(const std::string& stringID);
    std::vector<std::pair<geometry_msgs::msg::Pose, Task>> getPoseList(Mission mission);
    void makeTask(Mission mission);
    double calculate_palm_LM_distance_to_tool0(const geometry_msgs::msg::Point& palm_point);
    
    // NEW METHOD
    bool moveToSafeApproachPosition();

    geometry_msgs::msg::Pose getPose();
    std::pair<bool, double> checkPosition(const geometry_msgs::msg::Pose& current_local_pos, 
                                         const geometry_msgs::msg::Pose& target_position);

private:
    // RobotPoseUtility instead of MoveGroupInterface
    std::shared_ptr<RobotPoseUtility> pose_utility_;

    double euclideanDistance(double x1, double y1, double z1, double x2, double y2, double z2);
    std::pair<int, int> rotate90DegreesCounterClockwise(int x, int y);

    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr chessBoardPub;
    rclcpp::Publisher<checkers_msgs::msg::RobotMove>::SharedPtr robotMovePub;

    // Subscribers
    rclcpp::Subscription<checkers_msgs::msg::Board>::SharedPtr checkersBoardSub;
    void checkers_board_callback(const checkers_msgs::msg::Board::SharedPtr msg);

    rclcpp::Subscription<checkers_msgs::msg::Move>::SharedPtr checkersMoveSub;
    void checkers_move_callback(const checkers_msgs::msg::Move::SharedPtr msg);
    
    rclcpp::Subscription<checkers_msgs::msg::HandDetected>::SharedPtr handDetectedSub;
    void hand_detected_callback(const checkers_msgs::msg::HandDetected::SharedPtr msg);

    rclcpp::Subscription<leap_gesture_interface::msg::LeapFrame>::SharedPtr handDetectedLMSub;
    void hand_detected_lm_callback(const leap_gesture_interface::msg::LeapFrame::SharedPtr msg);
    
    rclcpp::Subscription<rcl_interfaces::msg::Log>::SharedPtr rosOutSub;
    void ros_out_callback(const rcl_interfaces::msg::Log::SharedPtr msg);

    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr palmPositionLMSub;
    void palm_position_lm_callback(const geometry_msgs::msg::Vector3::SharedPtr msg);

    // Service
    rclcpp::Client<gripper_srv::srv::GripperService>::SharedPtr gripperServiceClient;
    void handle_resume_movement(const std::shared_ptr<checkers_msgs::srv::ResumeMovement::Request> request,
                                std::shared_ptr<checkers_msgs::srv::ResumeMovement::Response> response);
    rclcpp::Service<checkers_msgs::srv::ResumeMovement>::SharedPtr serviceResumeMovement;

    // Variables
    std::vector<std::pair<geometry_msgs::msg::Pose, Task>> trajectory_list;
    geometry_msgs::msg::Pose tool0_pose;
    geometry_msgs::msg::Pose target_pose;
    int target_pose_index = 0;
    int trajectory_pose_index = 0;
    bool startProgram = true;
    std::unordered_map<int, bool> piecesInRviz;

    // Board calibration - measured real-world coordinates
    // Corner positions (center of corner squares):
    //   Bottom-right (7,7): X=0.8736, Y=0.4882 (closest to robot)
    //   Top-left (0,0):     X=0.4711, Y=0.0866 (farthest from robot)
    float square_size;      // Calculated from corner measurements
    float boardOffsetX;     // X position of square (0,0) center
    float boardOffsetY;     // Y position of square (0,0) center
    float zAttach;          // Height for attaching pieces (min safe Z = 0.155m)
    float zMoving;          // Safe height for moving above board
    float zMoveOffset;      // Movement offset after placing / grasping a checker
    float zSafeTransition;  // Safe transition height for movements between squares
    
    // Movement abort functionality
    std::atomic<bool> abortCurrentMovement{false};
    void abortAndClearMovement();

    visualization_msgs::msg::MarkerArray marker_array_fake_pieces;
    moveit_msgs::msg::CollisionObject collision_object;
    std::vector<Mission> targetPositions;
    bool doingTask = false;
    std::string whiteColorString = "white";
    std::string redColorString = "red";
    
    std::atomic<bool> isRobotMoving{false};
    std::atomic<bool> isStop{false};
    std::atomic<bool> hasValidTargetPose{false};  // Track if target_pose is valid
    std::atomic<bool> isRobotSendingHome{false};  // Track if doing HOME movement
    std::atomic<bool> isFinalHome{false};        // Track if this is the final HOME (uses moveToNamedTarget)
    geometry_msgs::msg::Pose homePose;           // Store HOME target pose for resumption
    std::unique_ptr<std::thread> moveThread;
    std::mutex moveMutex;  // Mutex for movement operations
    int removedPiecesCount = 0;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;

    geometry_msgs::msg::Pose currentPosition;
    int fileIndex = 34;
    std::string currentFileName = "robot_data_34.csv";

    void openGripper();
    void closeGripper();
};

#endif // ROBOT_CONTROL_NODE_HPP