#include <memory>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

class CameraPublisher : public rclcpp::Node
{
public:
  CameraPublisher()
  : Node("camera_publisher"), cap_(0)  // kamera /dev/video0
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(330),  // ~30 FPS
      std::bind(&CameraPublisher::timer_callback, this));

    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Nie udało się otworzyć kamery!");
    } else {
      RCLCPP_INFO(this->get_logger(), "Kamera uruchomiona, publikuje na /camera/image_raw");
    }
  }

private:
  void timer_callback()
  {
    cv::Mat frame;
    cap_ >> frame;
    if (frame.empty()) {
      RCLCPP_WARN(this->get_logger(), "Brak klatki z kamery");
      return;
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    publisher_->publish(*msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraPublisher>());
  rclcpp::shutdown();
  return 0;
}
