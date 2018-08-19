#include "../../include/gms_matcher_cluster_pose.h"


class image_to_gms
{

private:
	cv::Mat img1, img2; //video frames
	cv::Mat des1, des2; //descriptors
	

	bool frame_count;

	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

	
	float rotation_mean_degrees;
	pair<float, float> translation_mean_pix;

	ros::Publisher pub_pose;
	geometry_msgs::Pose2D pose;

public:
	
	image_to_gms(ros::NodeHandle &nh)
	{
		frame_count = false;
		pub_pose = nh.advertise<geometry_msgs::Pose2D> ("/geometry_msgs/Pose2D", 1);
	}

	void callback(const sensor_msgs::ImageConstPtr& msg);
	void publish_topic();

};

void image_to_gms::callback(const sensor_msgs::ImageConstPtr& msg)
{
	if(frame_count == true)
		img2 = cv_bridge::toCvShare(msg,"bgr8")->image;

	else
	{
		img1 = cv_bridge::toCvShare(msg,"bgr8")->image;
		frame_count = true;
		return;
	}

	

	imresize(img1, 480);
	imresize(img2, 480);

	vector<cv::KeyPoint> kp1, kp2; //keypoints
	vector<cv::DMatch> matches_all, matches_gms, matches_gms_inliers; // normal ORB matching, matching after applying GMS, matching after applying the cluster method in addition to GMS
	vector<bool> vbInliers;
	vector<uint> gms_cluster_outlier;
 
	orb->setFastThreshold(0);
	
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
	gms_matcher *gms = new gms_matcher(kp1, img1.size(), kp2, img2.size(), matches_all);
	gms->GetInlierMask(vbInliers, true, true); // 2nd argument is for scale and 3rd argument is for rotation

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
			matches_gms.push_back(matches_all[i]);
	}

	gms_cluster_outlier = gms->outlier_rejection(kp1, kp2, matches_gms);

	for(size_t i=0; i<matches_gms.size(); i++)
	{
		bool temp_condition = true;
		for(uint j=0; j<gms_cluster_outlier.size(); j++)
		{
			if(gms->cluster_index1.at<uint16_t>(kp1[matches_gms[i].queryIdx].pt.y,kp1[matches_gms[i].queryIdx].pt.x) == gms_cluster_outlier.at(j))
				temp_condition = false;
		}

		if(temp_condition!=false)
			matches_gms_inliers.push_back(matches_gms.at(i));
	}

	rotation_mean_degrees = gms->histogram_rotation_degrees();

	translation_mean_pix = gms->translation_histogram(kp1, kp2, matches_gms_inliers);

	cout<<endl<<"x: "<<translation_mean_pix.first<<endl<<"y: "<<translation_mean_pix.second<<endl<<"theta: "<<rotation_mean_degrees<<endl;

	publish_topic();
	
}

void image_to_gms::publish_topic()
{
	pose.x = translation_mean_pix.first;
	pose.y = translation_mean_pix.second;
	pose.theta = rotation_mean_degrees;

	pub_pose.publish(pose);
}