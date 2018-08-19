#include "gms_cluster_pose_pub.cpp"

using namespace std;

int main(int argc, char** argv)
{

	ros::init(argc, argv, "quadcop_pose_control_node");

	ros::NodeHandle nh;

	image_to_gms test (nh);

	image_transport::ImageTransport it(nh);

	image_transport::Subscriber img_sub = it.subscribe("/usb_cam/image_raw", 1, &image_to_gms::callback, &test);

	ros::spin();

	return 0;
}