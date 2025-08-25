cmake_minimum_required(VERSION 3.8)
project(cam_pubsub_yolo)

include_directories(${cv_bridge_INCLUDE_DIRS})


# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Publisher node
add_executable(camera_publisher src/camera_publisher.cpp)
ament_target_dependencies(camera_publisher
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

# Subscriber node
add_executable(camera_subscriber src/camera_subscriber.cpp)
ament_target_dependencies(camera_subscriber
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

ament_target_dependencies(camera_subscriber rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(camera_subscriber ${OpenCV_LIBS})

install(TARGETS
  camera_publisher
  camera_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
