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
      // Convert ROS image to OpenCV
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

      // Preprocess for YOLO
      const cv::Size inputSize(640, 640);
      cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, inputSize, cv::Scalar(), true, false);
      net_.setInput(blob);

      // Forward pass
      std::vector<cv::Mat> outputs;
      net_.forward(outputs, net_.getUnconnectedOutLayersNames());

      cv::Mat pred = outputs[0].reshape(1, outputs[0].total() / 85); // Nx85
      RCLCPP_INFO(this->get_logger(), "Detected objects: %d", pred.rows);

      // Scaling factors from 640x640 to original frame
      const float x_factor = float(frame.cols) / inputSize.width;
      const float y_factor = float(frame.rows) / inputSize.height;

      // Containers for NMS
      std::vector<cv::Rect> boxes;
      std::vector<float> confidences;
      std::vector<int> classIds;

      for (int i = 0; i < pred.rows; i++)
      {
        float conf = pred.at<float>(i, 4);
        if (conf < 0.5)
          continue; // skip low confidence

        // Bounding box coordinates
        float cx = pred.at<float>(i, 0);
        float cy = pred.at<float>(i, 1);
        float w = pred.at<float>(i, 2);
        float h = pred.at<float>(i, 3);

        int x = int((cx - w / 2.0f) * x_factor);
        int y = int((cy - h / 2.0f) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);

        // Class detection
        cv::Mat scores = pred.row(i).colRange(5, pred.cols);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);

        if (maxClassScore * conf < 0.05)
          continue; // combined threshold

        int classId = classIdPoint.x;
        boxes.push_back(cv::Rect(x, y, width, height));
        confidences.push_back(conf);
        classIds.push_back(classId);
      }

      // Apply Non-Maximum Suppression
      std::vector<int> indices;
      const float nmsThreshold = 0.4f;
      cv::dnn::NMSBoxes(boxes, confidences, 0.5f, nmsThreshold, indices);

      // Draw final detections
      for (int idx : indices)
      {
        cv::rectangle(frame, boxes[idx], cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, class_names[classIds[idx]], boxes[idx].tl(),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
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
  std::vector<std::string> class_names = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
      "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
      "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraSubscriber>());
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
