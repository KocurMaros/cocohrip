#include "robot_control/TestMovePublisher.hpp"
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

void displayMenu()
{
    std::cout << "\n╔════════════════════════════════════════════════════╗\n";
    std::cout << "║        Robot Control Test Menu                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  1 - Publish initial board state\n";
    std::cout << "  2 - Simple move (no capture)\n";
    std::cout << "  3 - Move with single capture\n";
    std::cout << "  4 - Move with multiple captures\n";
    std::cout << "  5 - Run sequence (board + simple move)\n";
    std::cout << "  6 - Move to board center and stay\n";
    std::cout << "  7 - Test 4 corner squares (returns HOME after each)\n";  // NEW
    std::cout << "  q - Quit\n";
    std::cout << "\n";
    std::cout << "Enter choice: ";
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto test_node = std::make_shared<TestMovePublisher>();
    
    // Spin in background thread
    std::thread spin_thread([test_node]() {
        rclcpp::spin(test_node);
    });
    
    std::cout << "\n╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   Robot Control Movement Test Publisher          ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Make sure robot_control_node is running!\n";
    std::cout << "Waiting for connections...\n";
    
    std::this_thread::sleep_for(2s);
    
    std::string choice;
    bool running = true;
    
    while (running && rclcpp::ok()) {
        displayMenu();
        std::getline(std::cin, choice);
        
        if (choice.empty()) continue;
        
        switch (choice[0]) {
            case '1':
                std::cout << "\n📋 Publishing initial board state...\n";
                test_node->publishTestBoard();
                std::this_thread::sleep_for(1s);
                std::cout << "✅ Board state published!\n";
                break;
                
            case '2':
                std::cout << "\n🤖 Publishing simple move command...\n";
                test_node->publishSimpleMove();
                std::cout << "✅ Move command sent! Watch the robot...\n";
                break;
                
            case '3':
                std::cout << "\n🎯 Publishing move with capture...\n";
                test_node->publishMoveWithCapture();
                std::cout << "✅ Capture move sent! Watch the robot...\n";
                break;
                
            case '4':
                std::cout << "\n⚡ Publishing move with multiple captures...\n";
                test_node->publishMultipleCaptures();
                std::cout << "✅ Multi-capture move sent! Watch the robot...\n";
                break;
                
            case '5':
                std::cout << "\n🔄 Running test sequence...\n";
                std::cout << "  Step 1: Publishing board state...\n";
                test_node->publishTestBoard();
                std::this_thread::sleep_for(3s);
                
                std::cout << "  Step 2: Publishing simple move...\n";
                test_node->publishSimpleMove();
                std::cout << "✅ Sequence started! Watch the robot complete the move...\n";
                break;
                
            case '6':
                std::cout << "\n🎯 Moving robot to board center...\n";
                test_node->publishMoveToBoardCenter();
                std::cout << "✅ Center move command sent! Robot will stay at center.\n";
                break;
                
            case '7':  // NEW
                std::cout << "\n🔲 Starting 4-corner squares test...\n";
                std::cout << "Robot will visit each corner and return HOME\n";
                std::cout << "This will take approximately 60 seconds...\n";
                test_node->publishCornerSquaresTest();
                std::cout << "✅ Corner test complete!\n";
                break;
                
            case 'q':
            case 'Q':
                std::cout << "\n👋 Exiting...\n";
                running = false;
                break;
                
            default:
                std::cout << "\n❌ Invalid choice! Please try again.\n";
                break;
        }
        
        if (running) {
            std::this_thread::sleep_for(500ms);
        }
    }
    
    rclcpp::shutdown();
    spin_thread.join();
    
    return 0;
}