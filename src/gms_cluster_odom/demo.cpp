// GridMatch.cpp : Defines the entry point for the console application.

#include "Header.h"
#include "gms_matcher.h"

int main()
{
	Mat img1 = imread("data/test6_c_pic1.jpg");
	Mat img2 = imread("data/test6_c_pic2.jpg");

	imresize(img1, 480);
	imresize(img2, 480);

	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms, matches_gms_inliers;

	Ptr<ORB> orb = ORB::create(1000);
	orb->setFastThreshold(0);
	
	if(img1.rows * img1.cols > 480 * 640 ){
		orb->setMaxFeatures(10000);
		orb->setFastThreshold(5);
	}
	
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher *gms = new gms_matcher(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms->GetInlierMask(vbInliers, true, true); // 2nd argument is for scale and 3rd argument is for rotation

	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	int match_count = 0;
	for(size_t i=0; i<matches_gms.size(); i++)
	{
		// cout<<"Matches ["<<i<<"]	-> Kp 1: "<<matches_gms[i].queryIdx<<"	- Kp 2: "<<matches_gms[i].trainIdx<<"	--K1C: "<<kp1[matches_gms[i].queryIdx].pt.x<<","<<kp1[matches_gms[i].queryIdx].pt.y<<"	- K2C: "<<kp2[matches_gms[i].trainIdx].pt.x<<","<<kp2[matches_gms[i].trainIdx].pt.y<<endl;
		match_count++;
	}

	cout<<"Match count:	"<<match_count<<endl;

	Mat cluster1 = gms->cluster(kp1, kp2, matches_gms);
	Mat cluster2 = gms->cluster_matching(kp1, kp2, matches_gms);
	gms->cluster_centroid(kp1, kp2, matches_gms);
	Mat drm = Mat::zeros(gms->cluster_count + 1, gms->cluster_count + 1, CV_32FC1);
	Mat cluster_rotation_deg = Mat::zeros(gms->cluster_count + 1, gms->cluster_count + 1, CV_32FC1);
	gms->distance_ratio_matrix(drm, cluster_rotation_deg);
	gms->histogram_distance_ratio(drm);
	vector<uint> gms_cluster_outlier = gms->outlier_rejection(drm);

	for(size_t i=0; i<matches_gms.size(); i++)
	{
		bool temp_condition = true;
		for(uint j=0; j<gms_cluster_outlier.size(); j++)
		{
			if(cluster1.at<uint16_t>(kp1[matches_gms[i].queryIdx].pt.y,kp1[matches_gms[i].queryIdx].pt.x) == gms_cluster_outlier.at(j))
				temp_condition = false;
		}

		if(temp_condition!=false)
			matches_gms_inliers.push_back(matches_gms.at(i));
	}

	gms->histogram_rotation_degrees(cluster_rotation_deg);
	gms->translation(kp1, kp2, matches_gms_inliers);
	gms->translation_histogram(matches_gms_inliers);
	// cout<<drm<<endl;
	Mat show = gms->DrawInlier(img1, img2, kp1, kp2, matches_gms, 2);
	imshow("show", show);

	Mat inliers_show = gms->DrawInlier(img1, img2, kp1, kp2, matches_gms_inliers, 2);
	imshow("inlier_show", inliers_show);

	fstream *op_dx = new fstream;
	op_dx->open("KeyPoint.txt", fstream::out);
	for (uint i = 0; i < matches_gms.size(); i++)
	{
		*op_dx<<"Matches ["<<i<<"]	-> Kp 1: "<<matches_gms[i].queryIdx<<"	- Kp 2: "<<matches_gms[i].trainIdx<<"	--K1C: "<<kp1[matches_gms[i].queryIdx].pt.x<<","<<kp1[matches_gms[i].queryIdx].pt.y<<"	- K2C: "<<kp2[matches_gms[i].trainIdx].pt.x<<","<<kp2[matches_gms[i].trainIdx].pt.y<<endl;
	}
	for (uint i = 0; i < vbInliers.size(); i++)
	{
		*op_dx<<"Index ["<<i<<"]	->	"<<vbInliers[i]<<endl;
	}
	*op_dx<<endl<<cluster1;
	op_dx->close();

	waitKey();

	return 0;
}