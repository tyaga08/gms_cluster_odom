#include "../../include/gms_matcher_cluster_pose.h"


class image_to_gms
{

private:
	cv::Mat img1, img2, previous_frame; //video frames after distortion correction

	cv::Mat des1, des2; //descriptors

	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

	float calibration_distance, dist_per_pixel, actual_height;

	bool first_frame;

	float rotation_mean_degrees;
	pair<float, float> translation_mean_pix;
	float trans_x, trans_y;

	ros::Publisher pub_pose, pub_transformation;
	geometry_msgs::Pose2D pose, transformation;

	int64_t step_count = 0;

public:
	
	image_to_gms(ros::NodeHandle &nh)
	{
		pub_pose = nh.advertise<geometry_msgs::Pose2D> ("/geometry_msgs/Pose2D", 1);
		pub_transformation = nh.advertise<geometry_msgs::Pose2D> ("/robot_camera/transformation",1);
		pose.x = 0;
		pose.y = 0;
		pose.theta = 0;
		transformation.x = 0;
		transformation.y = 0;
		transformation.theta = 0;
		calibration_distance = 1; //Mention the distance of the ground that is visible through the camera in the y direction at 1 metre height  
		actual_height = 0; //Replace zero with the data from the height measuring sensor
		dist_per_pixel = calibration_distance*actual_height/480;  //calibrated value; change the value according to the calibration

		first_frame = false;
	}

	vector<pair<float,float> > trans_array;

	void callback(const sensor_msgs::ImageConstPtr& msg);
	void publish_pose();
	void publish_transformation();

	~image_to_gms() {
	};
};

void image_to_gms::callback(const sensor_msgs::ImageConstPtr& msg)
{
	if(first_frame)
	{
		img2 = cv_bridge::toCvShare(msg,"bgr8")->image;
		img1 = previous_frame;
	}

	else
	{
		img1 = cv_bridge::toCvShare(msg,"bgr8")->image;
		previous_frame = img1;
		first_frame = true;
		return;
	}

	imresize(img1, 480);
	imresize(img2, 480);

	orb->setFastThreshold(0);
	
	vector<cv::KeyPoint> kp1, kp2; //keypoints
	vector<cv::DMatch> matches_all, matches_gms, matches_gms_inliers; // normal ORB matching, matching after applying GMS, matching after applying the cluster method in addition to GMS
	vector<bool> vbInliers;
	vector<uint> gms_cluster_outlier;

	if(img1.rows * img1.cols > 480 * 640 )
	{
		orb->setMaxFeatures(10000);
		orb->setFastThreshold(5);
	}
	
	orb->detectAndCompute(img1, cv::Mat(), kp1, des1);
	orb->detectAndCompute(img2, cv::Mat(), kp2, des2);

	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(des1, des2, matches_all);

	// GMS filter
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
	gms.GetInlierMask(vbInliers, true, true); // 2nd argument is for scale and 3rd argument is for rotation

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
			matches_gms.push_back(matches_all[i]);
	}

	gms_cluster_outlier = gms.outlier_rejection(kp1, kp2, matches_gms);

	for(size_t i=0; i<matches_gms.size(); i++)
	{
		bool temp_condition = true;
		for(uint j=0; j<gms_cluster_outlier.size(); j++)
		{
			if(gms.cluster_index1.at<uint16_t>(kp1[matches_gms[i].queryIdx].pt.y,kp1[matches_gms[i].queryIdx].pt.x) == gms_cluster_outlier.at(j))
				temp_condition = false;
		}

		if(temp_condition!=false)
			matches_gms_inliers.push_back(matches_gms.at(i));
	}

	rotation_mean_degrees = gms.histogram_rotation_degrees();

	translation_mean_pix = gms.translation_histogram(kp1, kp2, matches_gms_inliers);

	cout<<endl<<"x: "<<translation_mean_pix.first<<endl<<"y: "<<translation_mean_pix.second<<endl<<"theta: "<<rotation_mean_degrees<<endl;

	cv::Mat temp_trans = cv::Mat::eye(3,3,CV_32FC1);

	publish_pose();
	publish_transformation();
	
	cout<<"Step count = "<<++step_count<<endl;

	previous_frame = img2;
}

void image_to_gms::publish_pose() //publish pose in local frame
{

	if(!isnan(translation_mean_pix.first + translation_mean_pix.second + rotation_mean_degrees))
	{
		trans_x = translation_mean_pix.first*dist_per_pixel;
		trans_y = translation_mean_pix.second*dist_per_pixel;

		pose.x = pose.x + trans_x;
	
		pose.y = pose.y + trans_y;
	
		pose.theta += rotation_mean_degrees;
	}

	pub_pose.publish(pose);
}

void image_to_gms::publish_transformation() //publish pose in global frame
{
	if(!isnan(translation_mean_pix.first + translation_mean_pix.second + rotation_mean_degrees))
	{
		transformation.theta = pose.theta;
		transformation.x = transformation.x + cos((pose.theta - rotation_mean_degrees)*3.141592/180)*trans_x - sin((pose.theta - rotation_mean_degrees)*3.141592/180)*trans_y;
		transformation.y = transformation.y + cos((pose.theta - rotation_mean_degrees)*3.141592/180)*trans_y + sin((pose.theta - rotation_mean_degrees)*3.141592/180)*trans_x;

		trans_array.push_back(pair<float,float>(transformation.x, transformation.y));
	}																																																																								

	pub_transformation.publish(transformation);
}
