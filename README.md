GMS_CLUSTER_ODOM

This package is basically used to get odometry in a quadcopter using downward facing monocular camera. This will run in coordination to any depth/height measuring sensor.

Prerequisites:
OpenCV 3.0 or above

Steps to be followed:
1. Calibrate your camera. Change it's location mentioned in the camera_video.launch file.
2. Keep the camera at 1 metre from the ground. Measure the distance of the ground that is visible through the camera in the y direction at 1 metre height and replace the value of calibration_distance in gms_cluster_odom_pub.cpp (line 39)
3. Replace the zero value of actual_height in the line 40 of same file with the current readings of the depth/height measuring sensor.
4. Run catkin_make to build the package.
5. Open a terminal, go to the package and run the following command
	roslaunch launch/camera_video.launch
6. Open another terminal and run
	rosrun gms_cluster_odom quadcop_odom_node

Credits:
GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence.
https://github.com/JiawangBian/GMS-Feature-Matcher


