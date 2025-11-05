#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <termios.h>
#include <unistd.h>
#include <iostream>

class KeyInputPublisher : public rclcpp::Node
{
public:
  KeyInputPublisher()
  : Node("key_input_publisher")
  {
    pub_ = this->create_publisher<std_msgs::msg::String>("/key_input", 10);
    RCLCPP_INFO(this->get_logger(), "Key Input Publisher started.");
    read_loop();
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;

  void read_loop()
  {
    struct termios oldt, newt;
    char ch;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (rclcpp::ok())
    {
      if (read(STDIN_FILENO, &ch, 1) > 0)
      {
        auto msg = std_msgs::msg::String();
        msg.data = std::string(1, ch);
        pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Key sent: %c", ch);
      }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<KeyInputPublisher>();
  rclcpp::shutdown();
  return 0;
}
