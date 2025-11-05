#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/string.hpp>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <atomic>
#include <thread>
#include <gripper_srv/srv/gripper_service.hpp>

class UR5eMoveitController : public rclcpp::Node
{
public:
  UR5eMoveitController() : Node("ur5e_moveit_controller")
  {
    key_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/key_input", 10,
      std::bind(&UR5eMoveitController::onKeyInput, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "UR5e MoveIt controller node created. Call setup() after shared_ptr construction.");
  }

  void setup()
  {
    gripperServiceClient_ = this->create_client<gripper_srv::srv::GripperService>("gripper_service");
    while (!gripperServiceClient_->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_WARN(this->get_logger(), "Waiting for /gripper_service to become available...");
    }
    static const std::string PLANNING_GROUP = "ur_manipulator";
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), PLANNING_GROUP);
    move_group_->setPlanningTime(10.0);
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.1);

    RCLCPP_INFO(this->get_logger(), "UR5e MoveIt controller initialized.");
    std::thread(&UR5eMoveitController::executeScenarioLoop, this).detach();
  }

private:
  rclcpp::Client<gripper_srv::srv::GripperService>::SharedPtr gripperServiceClient_;

  void openGripper()
  {
    auto request = std::make_shared<gripper_srv::srv::GripperService::Request>();
    request->position = 0;  // 1 = open
    request->speed = 7;
    request->force = 5;
    auto result = gripperServiceClient_->async_send_request(request);
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  void closeGripper()
  {
    auto request = std::make_shared<gripper_srv::srv::GripperService::Request>();
    request->position = 110;  // 0 = close
    request->speed = 7;
    request->force = 5;
    auto result = gripperServiceClient_->async_send_request(request);
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr key_sub_;
  std::atomic_bool key_pressed_ = false;
  std::string latest_key_;

  void onKeyInput(const std_msgs::msg::String::SharedPtr msg)
  {
    latest_key_ = msg->data;
    key_pressed_ = true;
    RCLCPP_INFO(this->get_logger(), "Key input received: %s", latest_key_.c_str());
  }

  void waitForUserConfirmation(const std::string &message)
  {
    RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
    key_pressed_ = false;
    latest_key_.clear();
    while (rclcpp::ok() && !key_pressed_)
    {
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void waitForKey(char expected_key)
  {
    RCLCPP_INFO(this->get_logger(), "Press '%c' to continue...", expected_key);
    key_pressed_ = false;
    latest_key_.clear();
    while (rclcpp::ok())
    {
      if (key_pressed_ && latest_key_ == std::string(1, expected_key))
        break;
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    key_pressed_ = false;
    latest_key_.clear();
  }

  void executeScenarioLoop()
  {
    while (rclcpp::ok())
    {
      std::cout << "\nSelect scenario:\n"
                << "1. Scenario A (Home -> Pick -> Place)\n"
                << "2. Scenario B (Pick -> Home)\n"
                << "3. Scenario C (Home -> Place)\n"
                << "4. Exit\n"
                << "Press a key (1-4): " << std::endl;

      key_pressed_ = false;
      latest_key_.clear();
      while (rclcpp::ok() && !key_pressed_)
      {
        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }

      int choice = latest_key_.empty() ? -1 : latest_key_[0] - '0';
      if (choice < 1 || choice > 4)
      {
        std::cout << "Invalid choice. Please press 1-4.\n";
        continue;
      }

      RCLCPP_INFO(this->get_logger(), "User selected scenario %d", choice);

      if (choice == 4) break;

      switch (choice)
      {
        case 1:
          runScenarioA();
          break;
        case 2:
          runScenarioB();
          break;
        case 3:
          runScenarioC();
          break;
      }
    }

    RCLCPP_INFO(this->get_logger(), "Exiting scenario loop.");
  }

  void runScenarioA()
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 1.202;
    pose.position.y = 0.2971;
    pose.position.z = 0.4;
    waitForKey('p');
    moveToPlace(pose);
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 1.202;
    pose.position.y = 0.2971;
    pose.position.z = 0.36;
    waitForKey('p');
    moveToPlace(pose);
    closeGripper();
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 1.202;
    pose.position.y = 0.2971;
    pose.position.z = 0.4;
    moveToPlace(pose);
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.4;
    pose.position.y = 0.3;
    pose.position.z = 0.42;
    waitForKey('p');
    moveToPlace(pose);
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.4;
    pose.position.y = 0.3;
    pose.position.z = 0.36;
    waitForKey('p');
    moveToPlace(pose);
    openGripper();
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.4;
    pose.position.y = 0.3;
    pose.position.z = 0.4;
    moveToPlace(pose);

  }

  void runScenarioB()
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.48;
    pose.position.y = 0.32;
    pose.position.z = 0.2;
    waitForKey('p');
    moveToPlace(pose);
  }

  void runScenarioC()
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.4;
    pose.position.y = -0.2;
    pose.position.z = 0.3;
    waitForKey('p');
    if (moveToPlace(pose)) {
      openGripper();
      waitForUserConfirmation("Scenario C complete. Press any key to return to menu.");
    }
  }

  bool moveToHome()
  {
    std::vector<double> home_joint_positions = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};
    move_group_->setJointValueTarget(home_joint_positions);
    return (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool moveToPick()
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 0.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 1.0;
    pose.orientation.z = 0.0;
    pose.position.x = 0.4;
    pose.position.y = 0.2;
    pose.position.z = 0.3;
    move_group_->setPoseTarget(pose);
    return (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool moveToPlace(const geometry_msgs::msg::Pose &pose)
  {
    move_group_->setPoseTarget(pose);
    return (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UR5eMoveitController>();
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  node->setup();
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
