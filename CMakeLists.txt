cmake_minimum_required(VERSION 2.8.3)
project(gms_cluster_odom)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  roscpp
  rospy
  std_msgs
  cv_bridge
  eigen_conversions
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
INCLUDE_DIRS include
LIBRARIES gms_cluster_odom
CATKIN_DEPENDS geometry_msgs image_transport roscpp rospy std_msgs cv_bridge eigen_conversions
DEPENDS
)

include_directories(include
  ${catkin_INCLUDE_DIRS}/
  ${OpenCV_INCLUDE_DIRS}/
  ${PROJECT_SOURCE_DIR}/src/gms_cluster_odom/
  ${PROJECT_SOURCE_DIR}/include/
)

add_executable(quadcop_pose_control_node
src/quadcop_pose_control_node.cpp)

target_link_libraries (quadcop_pose_control_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(quadcop_odom_node
src/quadcop_odom_node.cpp)

target_link_libraries (quadcop_odom_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)