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
    net_ = cv::dnn::readNetFromONNX("/home/pawel/ros2_ws/models/yolo/yolov5s.onnx");
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw", 10,
        std::bind(&CameraSubscriber::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      auto frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
      // Preprocess for YOLO
      cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
      net_.setInput(blob);

      // Run inference
      std::vector<cv::Mat> outputs;
      net_.forward(outputs, net_.getUnconnectedOutLayersNames());

      cv::Mat pred = outputs[0]; // 1xNx85
      RCLCPP_INFO(this->get_logger(), "pred[0]size:%d", pred.size[0]);
      RCLCPP_INFO(this->get_logger(), "pred[0]size:%d", pred.size[1]);
      pred = pred.reshape(1, pred.total() / 85); // reshape do Nx85

      // Draw detections (simplified)
      RCLCPP_INFO(this->get_logger(), "pred[0]total:%ld", pred.total());
      RCLCPP_INFO(this->get_logger(), "pred[0]cols:%d", pred.cols);

      RCLCPP_INFO(this->get_logger(), "\nDetected objects:%d", pred.rows);

      float x_factor = float(frame.cols) / 640.0f;
      float y_factor = float(frame.rows) / 640.0f;

      for (int i = 0; i < pred.rows; i++)
      {
        float conf = pred.at<float>(i, 4);
        if (conf > 0.5)
        {
          float cx = pred.at<float>(i, 0);
          float cy = pred.at<float>(i, 1);
          float w = pred.at<float>(i, 2);
          float h = pred.at<float>(i, 3);

          int x = int((cx - w / 2) * x_factor);
          int y = int((cy - h / 2) * y_factor);
          int width = int(w * x_factor);
          int height = int(h * y_factor);
          RCLCPP_INFO(this->get_logger(),
                      "x:%d y:%d w:%f h:%f width:%d height:%d", x, y, w, h, width, height);
          cv::rectangle(frame, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);
        }
      }

      cv::imshow("YOLO Detection", frame);
      cv::waitKey(1);
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  cv::dnn::Net net_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraSubscriber>());
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
