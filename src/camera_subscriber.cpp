#include <memory>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

class CameraSubscriber : public rclcpp::Node
{
public:
  CameraSubscriber() : Node("camera_subscriber")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "camera/image_raw", 10,
      std::bind(&CameraSubscriber::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::imshow("Camera Frame", frame);
      cv::waitKey(1);
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraSubscriber>());
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
