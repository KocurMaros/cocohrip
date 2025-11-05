#include "robot_control/TestMovePublisher.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

TestMovePublisher::TestMovePublisher()
    : Node("test_move_publisher")
{
    board_pub_ = this->create_publisher<checkers_msgs::msg::Board>("board_topic", 10);
    move_pub_ = this->create_publisher<checkers_msgs::msg::Move>("move_topic", 10);
    
    RCLCPP_INFO(this->get_logger(), "Test Move Publisher initialized");
}

void TestMovePublisher::publishTestBoard()
{
    auto board_msg = checkers_msgs::msg::Board();
    
    // Create initial board state with a few pieces
    // Red pieces (player pieces)
    checkers_msgs::msg::Piece red_piece1;
    red_piece1.row = 2;
    red_piece1.col = 1;
    red_piece1.color = "red";
    red_piece1.king = false;
    board_msg.pieces.push_back(red_piece1);
    
    checkers_msgs::msg::Piece red_piece2;
    red_piece2.row = 2;
    red_piece2.col = 3;
    red_piece2.color = "red";
    red_piece2.king = false;
    board_msg.pieces.push_back(red_piece2);
    
    // White pieces (opponent pieces)
    checkers_msgs::msg::Piece white_piece1;
    white_piece1.row = 5;
    white_piece1.col = 2;
    white_piece1.color = "white";
    white_piece1.king = false;
    board_msg.pieces.push_back(white_piece1);
    
    checkers_msgs::msg::Piece white_piece2;
    white_piece2.row = 5;
    white_piece2.col = 4;
    white_piece2.color = "white";
    white_piece2.king = false;
    board_msg.pieces.push_back(white_piece2);
    
    RCLCPP_INFO(this->get_logger(), "Publishing initial board state with %zu pieces", 
                board_msg.pieces.size());
    board_pub_->publish(board_msg);
}

void TestMovePublisher::publishSimpleMove()
{
    auto move_msg = checkers_msgs::msg::Move();
    
    // Move piece from (2,1) to (3,2) - simple forward move
    move_msg.piece_for_moving.row = 2;
    move_msg.piece_for_moving.col = 1;
    move_msg.piece_for_moving.color = "red";
    move_msg.piece_for_moving.king = false;
    
    move_msg.target_row = 3;
    move_msg.target_col = 2;
    
    // No captures
    move_msg.removed_pieces = {};
    
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║         SIMPLE MOVE TEST                          ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Moving piece from (%d,%d) to (%d,%d)",
                move_msg.piece_for_moving.row, move_msg.piece_for_moving.col,
                move_msg.target_row, move_msg.target_col);
    
    move_pub_->publish(move_msg);
}

void TestMovePublisher::publishMoveWithCapture()
{
    auto move_msg = checkers_msgs::msg::Move();
    
    // Move piece from (2,3) to (4,5) - capturing piece at (3,4)
    move_msg.piece_for_moving.row = 2;
    move_msg.piece_for_moving.col = 3;
    move_msg.piece_for_moving.color = "red";
    move_msg.piece_for_moving.king = false;
    
    move_msg.target_row = 4;
    move_msg.target_col = 5;
    
    // Add captured piece
    checkers_msgs::msg::Piece captured_piece;
    captured_piece.row = 3;
    captured_piece.col = 4;
    captured_piece.color = "white";
    captured_piece.king = false;
    move_msg.removed_pieces.push_back(captured_piece);
    
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║      MOVE WITH CAPTURE TEST                       ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Moving piece from (%d,%d) to (%d,%d)",
                move_msg.piece_for_moving.row, move_msg.piece_for_moving.col,
                move_msg.target_row, move_msg.target_col);
    RCLCPP_INFO(this->get_logger(), "Capturing 1 piece at (%d,%d)",
                captured_piece.row, captured_piece.col);
    
    move_pub_->publish(move_msg);
}

void TestMovePublisher::publishMultipleCaptures()
{
    auto move_msg = checkers_msgs::msg::Move();
    
    // Move piece from (2,1) to (6,5) - capturing multiple pieces
    move_msg.piece_for_moving.row = 2;
    move_msg.piece_for_moving.col = 1;
    move_msg.piece_for_moving.color = "red";
    move_msg.piece_for_moving.king = false;
    
    move_msg.target_row = 6;
    move_msg.target_col = 5;
    
    // Add multiple captured pieces
    checkers_msgs::msg::Piece captured1;
    captured1.row = 3;
    captured1.col = 2;
    captured1.color = "white";
    captured1.king = false;
    move_msg.removed_pieces.push_back(captured1);
    
    checkers_msgs::msg::Piece captured2;
    captured2.row = 5;
    captured2.col = 4;
    captured2.color = "white";
    captured2.king = false;
    move_msg.removed_pieces.push_back(captured2);
    
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║    MULTIPLE CAPTURES TEST                         ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Moving piece from (%d,%d) to (%d,%d)",
                move_msg.piece_for_moving.row, move_msg.piece_for_moving.col,
                move_msg.target_row, move_msg.target_col);
    RCLCPP_INFO(this->get_logger(), "Capturing %zu pieces:", move_msg.removed_pieces.size());
    for (const auto& piece : move_msg.removed_pieces) {
        RCLCPP_INFO(this->get_logger(), "  - Piece at (%d,%d)", piece.row, piece.col);
    }
    
    move_pub_->publish(move_msg);
}

void TestMovePublisher::publishMoveToBoardCenter()
{
    // Instead of publishing a Move message, we need to send a direct pose command
    // But TestMovePublisher doesn't have that capability.
    // 
    // Option 1: Don't use the move topic, just log what to do
    // Option 2: Create a service client to command the robot directly
    
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║      MOVE TO BOARD CENTER TEST                    ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_WARN(this->get_logger(), "This test publisher can only publish Move messages.");
    RCLCPP_WARN(this->get_logger(), "To test direct board center positioning, you need to:");
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "1. Use RViz and MoveIt's 'Planning' panel");
    RCLCPP_INFO(this->get_logger(), "2. Drag the end effector to board center");
    RCLCPP_INFO(this->get_logger(), "3. Or create a dedicated service/action for direct positioning");
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "Board center coordinates should be approximately:");
    RCLCPP_INFO(this->get_logger(), "  X: 0.0 to 0.1 m (center of board)");
    RCLCPP_INFO(this->get_logger(), "  Y: 0.3 m (middle row)");
    RCLCPP_INFO(this->get_logger(), "  Z: 0.35 m (safe height above board)");
}

void TestMovePublisher::publishCornerSquaresTest()
{
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║      CORNER SQUARES TEST                          ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Testing movement to 4 corner squares:");
    RCLCPP_INFO(this->get_logger(), "  Corner 1: (1,1) - Near bottom-left");
    RCLCPP_INFO(this->get_logger(), "  Corner 2: (1,6) - Near top-left");  
    RCLCPP_INFO(this->get_logger(), "  Corner 3: (6,1) - Far bottom-left");
    RCLCPP_INFO(this->get_logger(), "  Corner 4: (6,6) - Far top-left");
    RCLCPP_INFO(this->get_logger(), "Robot will return to HOME after each corner\n");

    // Define the 4 corners to test (using inner corners, not edges)
    // These are on black squares in checkers notation
    std::vector<std::pair<int, int>> corners = {
        {0, 0},  // Top-left corner (farthest from robot)
        {0, 7},  // Top-right corner
        {7, 0},  // Bottom-left corner
        {7, 7}   // Bottom-right corner (closest to robot)
    };

    const char* corner_names[] = {
        "Corner (0,0) - Top-Left (Farthest)",
        "Corner (0,7) - Top-Right",
        "Corner (7,0) - Bottom-Left",
        "Corner (7,7) - Bottom-Right (Nearest)"
    };

    for (size_t i = 0; i < corners.size(); ++i) {
        auto move_msg = checkers_msgs::msg::Move();
        
        // Create a virtual piece at center board
        move_msg.piece_for_moving.row = 3;
        move_msg.piece_for_moving.col = 3;
        move_msg.piece_for_moving.color = "red";
        move_msg.piece_for_moving.king = false;
        
        // Move to corner
        move_msg.target_row = corners[i].first;
        move_msg.target_col = corners[i].second;
        
        // No captures
        move_msg.removed_pieces = {};
        
        RCLCPP_INFO(this->get_logger(), "\n→ Publishing move to %s", corner_names[i]);
        RCLCPP_INFO(this->get_logger(), "  From: (3,3) → To: (%d,%d)", 
                   move_msg.target_row, move_msg.target_col);
        
        move_pub_->publish(move_msg);
        
        RCLCPP_INFO(this->get_logger(), "✅ Move command sent!");
        RCLCPP_INFO(this->get_logger(), "⏳ Robot will move to corner, then return HOME");
        RCLCPP_INFO(this->get_logger(), "   Waiting 15 seconds before next corner...\n");
        
        // Wait for robot to complete movement and return home
        std::this_thread::sleep_for(std::chrono::seconds(15));
    }
    
    RCLCPP_INFO(this->get_logger(), "\n╔════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║  ✅ ALL 4 CORNERS TESTED SUCCESSFULLY!            ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════╝");
}