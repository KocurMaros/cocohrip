#include "robot_control/RobotControlNode.hpp"
#include "robot_control/RobotPoseUtility.hpp"

using namespace std::chrono_literals;

RobotControlNode::RobotControlNode()
        : Node("robot_control_node"),
          tf_buffer(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
          tf_listener(std::make_shared<tf2_ros::TransformListener>(*tf_buffer)) {

    // Publishers
    zMoveOffset = 0.10;
    chessBoardPub = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array", 10);
    robotMovePub = this->create_publisher<checkers_msgs::msg::RobotMove>("robot_move_topic", 10);

    // Subscribers
    checkersBoardSub = this->create_subscription<checkers_msgs::msg::Board>(
        "/board_topic", 10, std::bind(&RobotControlNode::checkers_board_callback, this, std::placeholders::_1));

    checkersMoveSub = this->create_subscription<checkers_msgs::msg::Move>(
        "/move_topic", 10, std::bind(&RobotControlNode::checkers_move_callback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "Subscribed to /move_topic - ready to receive move commands");
        
    handDetectedSub = this->create_subscription<checkers_msgs::msg::HandDetected>(
        "/hand_detected", 10, std::bind(&RobotControlNode::hand_detected_callback, this, std::placeholders::_1));

    handDetectedLMSub = this->create_subscription<std_msgs::msg::String>(
            "leap_gesture", 10, std::bind(&RobotControlNode::hand_detected_lm_callback, this, std::placeholders::_1));

    rosOutSub = this->create_subscription<rcl_interfaces::msg::Log>(
        "/rosout",10,std::bind(&RobotControlNode::ros_out_callback, this, std::placeholders::_1));

    palmPositionLMSub = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/leap_gesture/palm_position",10,std::bind(&RobotControlNode::palm_position_lm_callback, this, std::placeholders::_1));

    serviceResumeMovement = this->create_service<checkers_msgs::srv::ResumeMovement>(
        "resume_movement", std::bind(&RobotControlNode::handle_resume_movement, this, std::placeholders::_1, std::placeholders::_2));

    // Gripper service client
    gripperServiceClient = this->create_client<gripper_srv::srv::GripperService>("gripper_service");
    while (!gripperServiceClient->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "service not available, waiting again...");
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Real-world measured board corners (center of corner squares)
    // Closest to robot (bottom-right corner at position 7,7)
    float corner_near_x = 0.8736;
    float corner_near_y = 0.4882+0.01;
    
    // Farthest from robot (top-left corner at position 0,0)
    float corner_far_x = 0.4711;
    float corner_far_y = 0.0866+0.01;
    
    // Calculate square size from measured corners
    square_size = (corner_near_x - corner_far_x) / 7.0;  // Distance across 7 squares
    
    // Calculate board offsets (position of square 0,0 center)
    boardOffsetX = corner_far_x;
    boardOffsetY = corner_far_y;
    
    // Z heights
    zAttach = 0.155;  // End effector offset - minimum safe height
    zMoving = 0.20;   // Safe moving height above board (raised to 20cm to avoid hitting pieces)
    zSafeTransition = 0.25;  // Safe transition height for movements between squares (25cm)
    
    RCLCPP_INFO(this->get_logger(), "Board calibration:");
    RCLCPP_INFO(this->get_logger(), "  Calculated square size: %.4f m", square_size);
    RCLCPP_INFO(this->get_logger(), "  Board offset X: %.4f m", boardOffsetX);
    RCLCPP_INFO(this->get_logger(), "  Board offset Y: %.4f m", boardOffsetY);
    RCLCPP_INFO(this->get_logger(), "  Z attach height: %.4f m", zAttach);
    RCLCPP_INFO(this->get_logger(), "  Z moving height: %.4f m", zMoving);
}

RobotControlNode::~RobotControlNode() {
    if (moveThread && moveThread->joinable()) {
        moveThread->join();
    }
}

void RobotControlNode::initMoveGroup() {
    pose_utility_ = std::make_shared<RobotPoseUtility>(
        shared_from_this(),
        "ur5e_arm",
        "ur5e_tool0"
    );

    if (!pose_utility_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize RobotPoseUtility!");
        return;
    }

    pose_utility_->setVelocityScaling(0.8);  // Faster movement speed (80%)
    pose_utility_->setAccelerationScaling(0.3);  // Much faster acceleration

    RCLCPP_INFO(this->get_logger(), "RobotPoseUtility initialized successfully");

    getPose();
    openGripper();
    publishCheckerboard();
}

void RobotControlNode::openGripper() {
    RCLCPP_INFO(this->get_logger(), "Opening gripper");
    auto request = std::make_shared<gripper_srv::srv::GripperService::Request>();
    request->position = 0;
    request->speed = 5;
    request->force = 5;
    auto result = gripperServiceClient->async_send_request(request);
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

void RobotControlNode::closeGripper() {
    RCLCPP_INFO(this->get_logger(), "Closing gripper");
    auto request = std::make_shared<gripper_srv::srv::GripperService::Request>();
    request->position = 111;
    request->speed = 5;
    request->force = 5;
    auto result = gripperServiceClient->async_send_request(request);
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

bool RobotControlNode::moveToSafeApproachPosition() {
    RCLCPP_INFO(this->get_logger(), "Moving to safe approach position 20cm above chessboard center...");
    
    // Calculate center of chessboard
    float center_x = boardOffsetX + (3.5 * square_size);  // Middle of 8x8 board
    float center_y = boardOffsetY + (3.5 * square_size);
    float approach_z = 0.30;  // 20cm above board surface (board at ~0.10m)
    
    geometry_msgs::msg::Pose approach_pose;
    approach_pose.orientation.x = -0.0028119066264480352;
    approach_pose.orientation.y = 0.9999957084655762;
    approach_pose.orientation.z = -0.0007648332393728197;
    approach_pose.orientation.w = -0.00023792324645910412;
    approach_pose.position.x = center_x;
    approach_pose.position.y = center_y;
    approach_pose.position.z = approach_z;
    
    RCLCPP_INFO(this->get_logger(), 
                "Approach position: X=%.4f, Y=%.4f, Z=%.4f", 
                center_x, center_y, approach_z);
    
    // Move to approach position with longer planning time for large movements
    pose_utility_->setVelocityScaling(0.5);  // Slower for safety
    
    bool success = pose_utility_->moveToPose(approach_pose);
    
    if (success) {
        RCLCPP_INFO(this->get_logger(), "✅ Reached safe approach position");
        std::this_thread::sleep_for(std::chrono::seconds(2));  // Stabilize
    } else {
        RCLCPP_ERROR(this->get_logger(), "❌ Failed to reach approach position!");
        // Try again with more planning time
        RCLCPP_WARN(this->get_logger(), "Retrying with extended planning time...");
        success = pose_utility_->moveToPose(approach_pose);
        
        if (success) {
            RCLCPP_INFO(this->get_logger(), "✅ Reached approach position on retry");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    // Restore normal velocity
    pose_utility_->setVelocityScaling(0.8);
    
    return success;
}

void RobotControlNode::mainLoop() {
    getPose();
    
    if(isStop || isRobotMoving) {
        return;
    }
    
    if(targetPositions.empty() || doingTask) {
        return;
    }

    // Check if robot needs to move to approach position first
    if(target_pose_index == -1) {
        auto current_pose_opt = pose_utility_->getCurrentPose();
        
        if (current_pose_opt) {
            float board_center_x = boardOffsetX + (3.5 * square_size);
            float board_center_y = boardOffsetY + (3.5 * square_size);
            
            double distance_to_board = std::sqrt(
                std::pow(current_pose_opt->position.x - board_center_x, 2) +
                std::pow(current_pose_opt->position.y - board_center_y, 2)
            );
            
            if (distance_to_board > 0.5) {
                RCLCPP_WARN(this->get_logger(), 
                           "Robot is %.2f m from board center, moving to approach position first", 
                           distance_to_board);
                
                if (!moveToSafeApproachPosition()) {
                    RCLCPP_ERROR(this->get_logger(), 
                                "Failed to reach approach position! Cannot start board movements safely.");
                    targetPositions.clear();
                    return;
                }
            }
        }
    }

    if(trajectory_pose_index >= 0 && static_cast<std::size_t>(trajectory_pose_index) < trajectory_list.size()) {
        Task task = trajectory_list[trajectory_pose_index].second;
        if(task != Task::NONE) {
            Mission mission = targetPositions[target_pose_index];
            makeTask(mission);  
            return;
        }
        
        target_pose = trajectory_list[trajectory_pose_index].first;
        isRobotMoving = true;
        moveInThread(target_pose);
        trajectory_pose_index++;
        return;
    }

    // MODIFIED: Check if current mission trajectory is complete
    if(static_cast<std::size_t>(trajectory_pose_index) >= trajectory_list.size()) {
        
        // Check if there's a next mission
        if(static_cast<std::size_t>(target_pose_index + 1) < targetPositions.size()) {
            Mission current_mission = targetPositions[target_pose_index];
            Mission next_mission = targetPositions[target_pose_index + 1];
            
            // FIXED: Check if this is a PICK-PLACE pair (ATTACH followed by DETACH)
            bool is_pick_place_pair = (current_mission.task == Task::ATTACH && 
                                      next_mission.task == Task::DETACH);
            
            if (is_pick_place_pair) {
                // Don't go HOME yet, continue to DETACH (place)
                RCLCPP_INFO(this->get_logger(), 
                           "Pick complete at (%d,%d), moving to place at (%d,%d)...", 
                           current_mission.row, current_mission.col,
                           next_mission.row, next_mission.col);
                
                target_pose_index++;
                trajectory_pose_index = 0;
                trajectory_list = getPoseList(targetPositions[target_pose_index]);
                return;
            } else {
                // This is end of a complete move (DETACH), go HOME
                RCLCPP_INFO(this->get_logger(), 
                           "Move complete! Piece placed at (%d,%d). Returning to HOME...", 
                           current_mission.row, current_mission.col);
                
                target_pose_index++;
                trajectory_pose_index = 0;
                
                // Move to named "home" position
                isRobotSendingHome = true;
                isRobotMoving = true;
                
                if (moveThread && moveThread->joinable()) {
                    moveThread->join();
                }
                moveThread = std::make_unique<std::thread>([this]() {
                    RCLCPP_INFO(this->get_logger(), "Moving to named target: fake home");
                    geometry_msgs::msg::Pose current_pose = getPose();
                    current_pose.position.z += zMoveOffset;

                    bool success = pose_utility_->moveToPose(current_pose);
                    
                    if (success) {
                        RCLCPP_INFO(this->get_logger(), "✅ Reached 'fake home' position");
                    } else {
                        RCLCPP_ERROR(this->get_logger(), "❌ Failed to reach 'fake home' position!");
                    }
                    
                    isRobotMoving = false;
                    
                    if(isRobotSendingHome) {
                        fileIndex++;
                        currentFileName = "robot_data_" + std::to_string(fileIndex) + ".csv";
                        isRobotSendingHome = false;
                        
                        RCLCPP_INFO(this->get_logger(), "✅ Robot reached FAKE HOME position");
                        
                        // Publish robot move done message
                        auto message = checkers_msgs::msg::RobotMove();
                        message.robot_move_done = true;
                        robotMovePub->publish(message);
                        
                        // Prepare next trajectory if available
                        if(static_cast<std::size_t>(target_pose_index) < targetPositions.size() && target_pose_index >= 0) {
                            trajectory_list = getPoseList(targetPositions[target_pose_index]);
                            RCLCPP_INFO(this->get_logger(), "📋 Prepared trajectory for next move");
                        }
                    }
                });
                
                return;
            }
        }
        
        // All missions complete
        if(static_cast<std::size_t>(target_pose_index + 1) >= targetPositions.size()) {
            RCLCPP_INFO(this->get_logger(), "All moves complete! Final return to HOME...");
            
            target_pose_index = -1;
            trajectory_pose_index = 0;
            targetPositions.clear();
            trajectory_list.clear();
            
            // Final HOME
            isRobotSendingHome = true;
            isRobotMoving = true;
            
            if (moveThread && moveThread->joinable()) {
                moveThread->join();
            }
            moveThread = std::make_unique<std::thread>([this]() {
                RCLCPP_INFO(this->get_logger(), "Moving to final named target: home");
                geometry_msgs::msg::Pose current_pose = getPose();
                current_pose.position.z += zMoveOffset;
                bool success = pose_utility_->moveToPose(current_pose);

                success = pose_utility_->moveToNamedTarget("home") && success;
                
                if (success) {
                    RCLCPP_INFO(this->get_logger(), "✅ Reached final 'home' position");
                } else {
                    RCLCPP_ERROR(this->get_logger(), "❌ Failed to reach final 'home' position!");
                }
                
                isRobotMoving = false;
                
                if(isRobotSendingHome) {
                    fileIndex++;
                    currentFileName = "robot_data_" + std::to_string(fileIndex) + ".csv";
                    isRobotSendingHome = false;
                    
                    RCLCPP_INFO(this->get_logger(), "✅ All tasks complete! Robot at HOME");
                    
                    auto message = checkers_msgs::msg::RobotMove();
                    message.robot_move_done = true;
                    robotMovePub->publish(message);
                }
            });
        }
    }
}

void RobotControlNode::makeTask(Mission mission) {
    doingTask = true;
    RCLCPP_INFO(this->get_logger(), "Making task");

    Task task = mission.task;
    int row = mission.row;
    int col = mission.col;
    std::string color = mission.color;

    std::string pieceID = "piece" + std::to_string(row) + std::to_string(col);

    if(task == Task::ATTACH) {
        removeFakePiece(pieceID);
        createPiece(row, col);
        // attachPiece();
        closeGripper();
    }
    else {
        // detachPiece();
        removePiece();
        createFakePieceWithColor(pieceID, row, col, color);
        chessBoardPub->publish(marker_array_fake_pieces);
        openGripper();
    }

    targetPositions[target_pose_index].task = Task::NONE;
    trajectory_list[trajectory_pose_index].second = Task::NONE;
    doingTask = false;
}

std::pair<bool, double> RobotControlNode::checkPosition(const geometry_msgs::msg::Pose& current_local_pos, 
                                                        const geometry_msgs::msg::Pose& target_position) {
    double threshold = 0.05;

    double current_x = current_local_pos.position.x;
    double current_y = current_local_pos.position.y;
    double current_z = current_local_pos.position.z;
    if (current_z < 0) {
        current_z = 0.0;
    }

    double target_x = target_position.position.x;
    double target_y = target_position.position.y;
    double target_z = target_position.position.z;

    double distance = euclideanDistance(current_x, current_y, current_z, target_x, target_y, target_z);

    if (distance < threshold) {
        return std::make_pair(true, distance);
    } else {
        return std::make_pair(false, distance);
    }
}

double RobotControlNode::euclideanDistance(double x1, double y1, double z1, double x2, double y2, double z2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
}

geometry_msgs::msg::Pose RobotControlNode::getPose() {
    auto current_pose_opt = pose_utility_->getCurrentPose();
    if (current_pose_opt) {
        currentPosition = *current_pose_opt;
    } else {
        RCLCPP_WARN(this->get_logger(), "Failed to get current pose");
    }
    return currentPosition;
}

void RobotControlNode::move(geometry_msgs::msg::Pose targetPose) {
    isRobotMoving = true;
    
    bool success = pose_utility_->moveToPose(targetPose);
    
    if (!success) {
        RCLCPP_ERROR(this->get_logger(), "Movement failed! Retrying...");
        success = pose_utility_->moveToPose(targetPose);
        if (!success) {
            RCLCPP_ERROR(this->get_logger(), "Movement failed after retry!");
            isRobotMoving = false;
        }
    }
}

void RobotControlNode::moveInThread(geometry_msgs::msg::Pose targetPose) {
    RCLCPP_INFO(this->get_logger(), 
                   "Moving to position: [%f, %f, %f]", 
                   targetPose.position.x, targetPose.position.y, targetPose.position.z);
    if (moveThread && moveThread->joinable()) {
        moveThread->join();
    }
    moveThread = std::make_unique<std::thread>(&RobotControlNode::move, this, targetPose);
}

void RobotControlNode::stop() {
    isStop = true;
    RCLCPP_WARN(this->get_logger(), "Stop requested - current movement will complete");
}

void RobotControlNode::handle_resume_movement(
    const std::shared_ptr<checkers_msgs::srv::ResumeMovement::Request> request,
    std::shared_ptr<checkers_msgs::srv::ResumeMovement::Response> response) {
    isStop = false;
    RCLCPP_INFO(this->get_logger(), "Resuming robot movement.");
    response->success = true;
    moveInThread(target_pose);
}

void RobotControlNode::ros_out_callback(const rcl_interfaces::msg::Log::SharedPtr msg) {
    if(isRobotMoving) {
        if (msg->msg.find("Goal reached, success!") != std::string::npos ||
            msg->msg.find("Controller 'joint_trajectory_controller' successfully finished") != std::string::npos ||
            msg->msg.find("Completed trajectory execution with status SUCCEEDED") != std::string::npos ||
            msg->msg.find("Solution was found and executed.") != std::string::npos ||
            msg->msg.find("Plan and Execute request complete!") != std::string::npos ||
            msg->msg.find("Movement successful!") != std::string::npos)
        {
            isRobotMoving = false;
            
            if(isRobotSendingHome) {
                fileIndex++;
                currentFileName = "robot_data_" + std::to_string(fileIndex) + ".csv";
                isRobotSendingHome = false;
                
                RCLCPP_INFO(this->get_logger(), "✅ Robot reached HOME position");
                
                // Publish robot move done message
                auto message = checkers_msgs::msg::RobotMove();
                message.robot_move_done = true;
                robotMovePub->publish(message);
                
                // If there are more missions, prepare next trajectory
                if(static_cast<std::size_t>(target_pose_index) < targetPositions.size() && target_pose_index >= 0) {
                    trajectory_list = getPoseList(targetPositions[target_pose_index]);
                    RCLCPP_INFO(this->get_logger(), "📋 Prepared trajectory for mission %d", target_pose_index);
                }
            }
        }

        if (msg->msg.find("Unable to solve the planning problem") != std::string::npos ||
            msg->msg.find("Motion plan could not be computed") != std::string::npos ||
            msg->msg.find("Failed to find a motion plan") != std::string::npos ||
            msg->msg.find("Planning failed!") != std::string::npos ||
            msg->msg.find("Timed out") != std::string::npos ||
            msg->msg.find("TIMED_OUT") != std::string::npos) {
            RCLCPP_WARN(this->get_logger(), "Planning failed, retrying...");
            moveInThread(target_pose);
        }
    }
}

void RobotControlNode::palm_position_lm_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
    geometry_msgs::msg::PointStamped palm_point;
    palm_point.point.x = msg->x;
    palm_point.point.y = msg->y;
    palm_point.point.z = msg->z;

    double distance = calculate_palm_LM_distance_to_tool0(palm_point.point);
    
    if(distance < 0.275 && isRobotMoving) {
        stop();
    } else if(isStop){
        isStop = false;
        moveInThread(target_pose);
    }
}

double RobotControlNode::calculate_palm_LM_distance_to_tool0(const geometry_msgs::msg::Point& palm_point) {
    double distance = std::sqrt(
        std::pow(currentPosition.position.x - palm_point.x, 2) +
        std::pow(currentPosition.position.y - palm_point.y, 2) +
        std::pow((currentPosition.position.z - 0.013) - palm_point.z, 2)
    );

    if(isRobotMoving && !isStop) {
        std::ofstream outfile;
        outfile.open("/home/collab/amavet_ws/src/robot_control/src/" + currentFileName, std::ios_base::app);
        if (outfile.is_open()) {
            outfile << currentPosition.position.x << ","
                    << currentPosition.position.y << ","
                    << currentPosition.position.z << ","
                    << palm_point.x << ","
                    << palm_point.y << ","
                    << palm_point.z << ","
                    << (distance < 0.275 && isRobotMoving ? distance : 0) << std::endl;
            outfile.close();
        }
    }

    return distance;
}

void RobotControlNode::hand_detected_callback(const checkers_msgs::msg::HandDetected::SharedPtr msg) {
    if(isRobotMoving) {
        stop();
    }
}

void RobotControlNode::hand_detected_lm_callback(const std_msgs::msg::String::SharedPtr msg) {
    // Implementation if needed
}

void RobotControlNode::abortAndClearMovement() {
    RCLCPP_WARN(this->get_logger(), "⚠️ ABORTING current movement for new request!");
    
    // Signal abort
    abortCurrentMovement.store(true);
    
    // Stop current movement via MoveIt
    if (pose_utility_ && pose_utility_->isInitialized()) {
        // MoveGroup stop will be handled by the abort flag
    }
    
    // Wait for movement thread to finish
    if (moveThread && moveThread->joinable()) {
        moveThread->join();
    }
    
    // Clear state
    isRobotMoving = false;
    isStop = false;
    doingTask = false;
    abortCurrentMovement.store(false);
    
    RCLCPP_INFO(this->get_logger(), "✅ Previous movement aborted, ready for new command");
}

void RobotControlNode::checkers_move_callback(const checkers_msgs::msg::Move::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "\n========================================");
    RCLCPP_INFO(this->get_logger(), "RECEIVED MOVE COMMAND!");
    RCLCPP_INFO(this->get_logger(), "  From: (%d, %d) -> To: (%d, %d)", 
                msg->piece_for_moving.row, msg->piece_for_moving.col,
                msg->target_row, msg->target_col);
    RCLCPP_INFO(this->get_logger(), "  Removed pieces: %zu", msg->removed_pieces.size());
    RCLCPP_INFO(this->get_logger(), "========================================\n");
    
    // Add safety check at the start
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "Received null move message!");
        return;
    }
    
    // ABORT current movement if robot is moving
    if (isRobotMoving || !targetPositions.empty()) {
        abortAndClearMovement();
    }
    
    // Clear existing data safely
    targetPositions.clear();
    trajectory_list.clear();
    target_pose_index = -1;
    trajectory_pose_index = 0;

    auto [startRow, startCol] = rotate90DegreesCounterClockwise(msg->piece_for_moving.row, msg->piece_for_moving.col);
    
    // Validate coordinates before adding
    if (startRow < 0 || startRow > 9 || startCol < 0 || startCol > 9) {
        RCLCPP_ERROR(this->get_logger(), "Invalid start coordinates: [%d, %d]", startRow, startCol);
        return;
    }
    
    targetPositions.push_back(Mission(startRow, startCol, redColorString, Task::ATTACH));

    auto [targetRow, targetCol] = rotate90DegreesCounterClockwise(msg->target_row, msg->target_col);
    
    // Validate target coordinates
    if (targetRow < 0 || targetRow > 9 || targetCol < 0 || targetCol > 9) {
        RCLCPP_ERROR(this->get_logger(), "Invalid target coordinates: [%d, %d]", targetRow, targetCol);
        targetPositions.clear();
        return;
    }
    
    targetPositions.push_back(Mission(targetRow, targetCol, redColorString, Task::DETACH));

    // Process removed pieces with validation
    for (const auto& piece : msg->removed_pieces) {
        auto [row, col] = rotate90DegreesCounterClockwise(piece.row, piece.col);
        
        // Fixed: Create columns of 4 pieces each on the side
        // Column 0: pieces 0-3, Column 1: pieces 4-7, Column 2: pieces 8-11
        int storage_col = removedPiecesCount / 4;  // New column every 4 pieces
        int storage_row = 8 + (removedPiecesCount % 4);  // Row within column (0-3)
        
        // Add bounds checking for storage positions (max 3 columns of 4 pieces = 12 pieces)
        if (storage_col > 2) {
            RCLCPP_ERROR(this->get_logger(), "Storage area full! Cannot store more pieces.");
            continue;
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "Storage calculation: piece #%d -> column %d, row offset %d", 
                   removedPiecesCount, storage_col, storage_row - 8);
        
        removedPiecesCount++;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Storing removed piece from [%d, %d] at storage position [%d, %d]", 
                   row, col, storage_row, storage_col);

        targetPositions.push_back(Mission(row, col, whiteColorString, Task::ATTACH));
        targetPositions.push_back(Mission(storage_row+1, storage_col, whiteColorString, Task::DETACH));
    }
    
    // Prepare first trajectory
    if (!targetPositions.empty()) {
        target_pose_index = 0;
        trajectory_pose_index = 0;
        trajectory_list = getPoseList(targetPositions[0]);
        RCLCPP_INFO(this->get_logger(), "Prepared trajectory with %zu poses", trajectory_list.size());
    }
}
std::vector<std::pair<geometry_msgs::msg::Pose, Task>> RobotControlNode::getPoseList(Mission mission) {
    trajectory_pose_index = 0;
    std::vector<std::pair<geometry_msgs::msg::Pose, Task>> poses;

    float posX = (mission.row * square_size) + boardOffsetX;
    float posY = (mission.col * square_size) + boardOffsetY;
    
    // SAFETY CHECK: Validate coordinates are within reasonable bounds
    float min_x = boardOffsetX - square_size;  // Allow 1 square beyond board
    float max_x = boardOffsetX + (9 * square_size);  // Board + 1 extra square
    float min_y = boardOffsetY - square_size;
    float max_y = boardOffsetY + (9 * square_size);
    
    if (posX < min_x || posX > max_x || posY < min_y || posY > max_y) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Target position OUT OF BOUNDS! Row=%d, Col=%d -> X=%.4f, Y=%.4f",
                    mission.row, mission.col, posX, posY);
        RCLCPP_ERROR(this->get_logger(), 
                    "Valid range: X[%.4f to %.4f], Y[%.4f to %.4f]",
                    min_x, max_x, min_y, max_y);
        
        // Clamp to valid range
        posX = std::max(min_x, std::min(max_x, posX));
        posY = std::max(min_y, std::min(max_y, posY));
        
        RCLCPP_WARN(this->get_logger(), "Clamped to: X=%.4f, Y=%.4f", posX, posY);
    }

    // Check if we need an intermediate high waypoint
    auto current_pose_opt = pose_utility_->getCurrentPose();
    
    if (current_pose_opt) {
        double horizontal_distance = std::sqrt(
            std::pow(current_pose_opt->position.x - posX, 2) +
            std::pow(current_pose_opt->position.y - posY, 2)
        );
        
        RCLCPP_INFO(this->get_logger(), 
                   "Moving from [%.4f, %.4f] to [%.4f, %.4f], distance: %.4f m",
                   current_pose_opt->position.x, current_pose_opt->position.y,
                   posX, posY, horizontal_distance);
        
        // For any movement, first lift to safe transition height (25cm) to avoid pieces
        if (current_pose_opt->position.z < zSafeTransition) {
            geometry_msgs::msg::Pose lift_pose;
            lift_pose.orientation.x = -0.0028119066264480352;
            lift_pose.orientation.y = 0.9999957084655762;
            lift_pose.orientation.z = -0.0007648332393728197;
            lift_pose.orientation.w = -0.00023792324645910412;
            lift_pose.position.x = current_pose_opt->position.x;
            lift_pose.position.y = current_pose_opt->position.y;
            lift_pose.position.z = zSafeTransition;  // Lift to 25cm first
            poses.push_back(std::make_pair(lift_pose, Task::NONE));
            
            RCLCPP_INFO(this->get_logger(), "Added lift waypoint to Z=%.2f", zSafeTransition);
        }
        
        // If moving more than 15cm horizontally, add intermediate waypoint at safe height
        if (horizontal_distance > 0.15) {
            geometry_msgs::msg::Pose travel_waypoint;
            travel_waypoint.orientation.x = -0.0028119066264480352;
            travel_waypoint.orientation.y = 0.9999957084655762;
            travel_waypoint.orientation.z = -0.0007648332393728197;
            travel_waypoint.orientation.w = -0.00023792324645910412;
            travel_waypoint.position.x = posX;
            travel_waypoint.position.y = posY;
            travel_waypoint.position.z = zSafeTransition;  // Travel at 25cm height
            poses.push_back(std::make_pair(travel_waypoint, Task::NONE));
            
            RCLCPP_INFO(this->get_logger(), 
                       "Added travel waypoint at Z=%.2f (distance: %.2f m)", zSafeTransition, horizontal_distance);
        }
    }

    // First pose: above target square at moving height
    geometry_msgs::msg::Pose pose1;
    pose1.orientation.x = -0.0028119066264480352;
    pose1.orientation.y = 0.9999957084655762;
    pose1.orientation.z = -0.0007648332393728197;
    pose1.orientation.w = -0.00023792324645910412;
    pose1.position.x = posX;
    pose1.position.y = posY;
    pose1.position.z = zMoving;
    poses.push_back(std::make_pair(pose1, Task::NONE));
// 
    // // Second pose: down at attach height
    geometry_msgs::msg::Pose pose2;
    pose2.orientation = pose1.orientation;
    pose2.position.x = posX;
    pose2.position.y = posY;
    pose2.position.z = zAttach - 0.0107;
    poses.push_back(std::make_pair(pose2, Task::NONE));
// 
    // // Third pose: back up to moving height
    geometry_msgs::msg::Pose pose3;
    pose3.orientation = pose1.orientation;
    pose3.position.x = posX;
    pose3.position.y = posY;
    pose3.position.z = zMoving;
    // poses.push_back(std::make_pair(pose1, Task::NONE));
    poses.push_back(std::make_pair(pose3, mission.task));

    return poses;
}

void RobotControlNode::checkers_board_callback(const checkers_msgs::msg::Board::SharedPtr msg) {
    removeAllFakePieces();

    for (const auto& piece : msg->pieces) {
        auto [row, col] = rotate90DegreesCounterClockwise(piece.row, piece.col);
        std::string pieceID = "piece" + std::to_string(row) + std::to_string(col);
        createFakePieceWithColor(pieceID, row, col, piece.color);
    }

    chessBoardPub->publish(marker_array_fake_pieces);
    
    if(startProgram) {
        startProgram = false;
    }
}

std::tuple<float, float, float> RobotControlNode::getColorFromName(const std::string& colorName) {
    if (colorName == "white") {
        return std::make_tuple(1.0f, 1.0f, 1.0f);
    } else if (colorName == "red") {
        return std::make_tuple(1.0f, 0.0f, 0.0f);
    }
    return std::make_tuple(0.0f, 0.0f, 0.0f);
}

void RobotControlNode::createFakePieceWithColor(const std::string& object_id, int row, int col, 
                                                const std::string& colorName) {
    RCLCPP_INFO(this->get_logger(), "Creaing fake piece");\

    int objectIDLong = convertStringToInt(object_id);
    piecesInRviz[objectIDLong] = true;

    std::tuple<float, float, float> color = getColorFromName(colorName);
    
    visualization_msgs::msg::Marker fakePiece;
    fakePiece.header.frame_id = "base_link";
    fakePiece.id = objectIDLong;
    fakePiece.type = visualization_msgs::msg::Marker::CYLINDER;
    fakePiece.action = visualization_msgs::msg::Marker::ADD;

    fakePiece.pose.position.x = (row * square_size) + boardOffsetX;
    fakePiece.pose.position.y = (col * square_size) + boardOffsetY;
    fakePiece.pose.position.z = 0.0075;
    fakePiece.pose.orientation.w = 1.0;

    fakePiece.scale.x = 0.028;
    fakePiece.scale.y = 0.028;
    fakePiece.scale.z = 0.005;

    fakePiece.color.r = std::get<0>(color);
    fakePiece.color.g = std::get<1>(color);
    fakePiece.color.b = std::get<2>(color);
    fakePiece.color.a = 1.0;

    marker_array_fake_pieces.markers.push_back(fakePiece);
}

int RobotControlNode::convertStringToInt(const std::string& stringID){
    long long concatenatedNumber = 0;
    for (char c : stringID) {
        concatenatedNumber = concatenatedNumber * 1000 + static_cast<int>(c);
    }
    return concatenatedNumber;
}

void RobotControlNode::removeAllFakePieces() {
    for (auto& marker : marker_array_fake_pieces.markers) {
        int markerID = marker.id;
        if (piecesInRviz.find(markerID) != piecesInRviz.end() && piecesInRviz[markerID]) {
            marker.action = visualization_msgs::msg::Marker::DELETE;
        }
    }

    chessBoardPub->publish(marker_array_fake_pieces);
    piecesInRviz.clear();
    marker_array_fake_pieces.markers.clear();
}

void RobotControlNode::removeFakePiece(const std::string& object_id) {
    RCLCPP_INFO(this->get_logger(), "Removing fake piece");\
    int objectIDLong = convertStringToInt(object_id);
    visualization_msgs::msg::MarkerArray updated_marker_array;

    for (auto& marker : marker_array_fake_pieces.markers) {
        if (marker.id != objectIDLong) {
            updated_marker_array.markers.push_back(marker);
        } else {
            marker.action = visualization_msgs::msg::Marker::DELETE;
            if (piecesInRviz.find(objectIDLong) != piecesInRviz.end()) {
                piecesInRviz.erase(objectIDLong);
            }
        }
    }

    chessBoardPub->publish(marker_array_fake_pieces);
    marker_array_fake_pieces = updated_marker_array;
}

void RobotControlNode::createPiece(int row, int col) {
    RCLCPP_INFO(this->get_logger(), "Creating piece");\
    collision_object = moveit_msgs::msg::CollisionObject();
    collision_object.header.frame_id = "base_link";
    collision_object.id = "collisionObjectID";
    
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.CYLINDER;
    primitive.dimensions.resize(2);
    primitive.dimensions[primitive.CYLINDER_HEIGHT] = 0.005;
    primitive.dimensions[primitive.CYLINDER_RADIUS] = 0.014;

    float posX = (row * square_size) + boardOffsetX;
    float posY = (col * square_size) + boardOffsetY;

    geometry_msgs::msg::Pose cylinder_pose;
    cylinder_pose.orientation.w = 1.0;
    cylinder_pose.position.x = posX;
    cylinder_pose.position.y = posY;
    cylinder_pose.position.z = 0.0075;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(cylinder_pose);
    collision_object.operation = collision_object.ADD;

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface.applyCollisionObject(collision_object);
}

void RobotControlNode::removePiece() {
    RCLCPP_INFO(this->get_logger(), "Removing piece");\
    collision_object.operation = collision_object.REMOVE;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface.applyCollisionObject(collision_object);
}

void RobotControlNode::attachPiece() {
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    RCLCPP_INFO(this->get_logger(), "Attach piece using planning scene interface");
}

void RobotControlNode::detachPiece() {
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    RCLCPP_INFO(this->get_logger(), "Detach piece using planning scene interface");
}

void RobotControlNode::publishCheckerboard() {
    visualization_msgs::msg::MarkerArray marker_array;
    int rows = 8;
    int cols = 8;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto [rotatedRow, rotatedCol] = rotate90DegreesCounterClockwise(row, col);

            visualization_msgs::msg::Marker square_marker;
            square_marker.header.frame_id = "base_link";
            square_marker.type = visualization_msgs::msg::Marker::CUBE;
            square_marker.action = visualization_msgs::msg::Marker::ADD;

            square_marker.pose.position.x = (rotatedRow * square_size) + boardOffsetX;
            square_marker.pose.position.y = (rotatedCol * square_size) + boardOffsetY;
            square_marker.pose.position.z = 0.0;
            square_marker.pose.orientation.w = 1.0;

            square_marker.scale.x = square_size;
            square_marker.scale.y = square_size;
            square_marker.scale.z = 0.01;

            if ((row + col) == 0) {
                square_marker.color.r = 0.0;
                square_marker.color.g = 1.0;
                square_marker.color.b = 0.0;
            } else if ((rotatedRow + rotatedCol) % 2 == 0) {
                square_marker.color.r = 1.0;
                square_marker.color.g = 1.0;
                square_marker.color.b = 1.0;
            } else {
                square_marker.color.r = 0.0;
                square_marker.color.g = 0.0;
                square_marker.color.b = 0.0;
            }

            square_marker.color.a = 1.0;
            square_marker.id = rotatedRow * cols + rotatedCol;
            marker_array.markers.push_back(square_marker);
        }
    }

    chessBoardPub->publish(marker_array);
}

std::pair<int, int> RobotControlNode::rotate90DegreesCounterClockwise(int x, int y) {
    // FIXED: Rotate 180 degrees instead of 90 degrees counter-clockwise
    // This flips both X and Y to match the physical board orientation
    // Original checkers coords (0,0) = top-left should map to physical far corner
    // Original checkers coords (7,7) = bottom-right should map to physical near corner
    
    int rotated_x = 7 - x;  // Flip X axis
    int rotated_y = 7 - y;  // Flip Y axis

    return {y, rotated_x};
}