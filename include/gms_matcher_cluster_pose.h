#pragma once
#include "Header.h"

#define THRESH_FACTOR 6

// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class gms_matcher
{
public:
	// OpenCV cv::KeyPoints & Correspond Image cv::Size & Nearest Neighbor Matches 
	gms_matcher(const vector<cv::KeyPoint> &vkp1, const cv::Size size1, const vector<cv::KeyPoint> &vkp2, const cv::Size size2, const vector<cv::DMatch> &vDMatches) {
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
		mGridSizeLeft = cv::Size(20, 20);
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neihbor of left grid 
		mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);

		// Initialize the cluster components
		cluster_index1 = cv::Mat::zeros(size1.height, size1.width, CV_16UC1);
		cluster_index2 = cv::Mat::zeros(size1.height, size1.width, CV_16UC1);

		cluster_count = 0;
		cluster_mask = 10;


		hist_min = 1000.0;
		hist_max = 0.0;
		hist_mode = 0.0;
		hist_threshold = 0.15;
		hist_mode_idx = 0;

		// cout<<"a"<<a<<endl;
	};

	~gms_matcher() {
	};


private:

	// Normalized Points
	vector <cv::Point2f> mvP1, mvP2;

	// Matches
	vector <pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	cv::Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	cv::Mat mMotionStatistics;

	// 
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	//
	cv::Mat mGridNeighborLeft;
	cv::Mat mGridNeighborRight;

public:
	// For clustering
	cv::Mat cluster_index1, cluster_index2; //cluster numbering
	uint16_t cluster_count, temp_idx; //required for counting the number of clusters
	int cluster_mask; //the pixels about a keypoint, around which the algorithm would search for another keypoint to add in the same cluster 
	float dist_idx;
	vector<pair<float,float> > centroid_query, centroid_train; // cluster centroids coordinates of the first and the second image

	//Distance ratio matrix
	float ratio_mean;// the average distance ratio
	cv::Mat cluster_rotation_deg, drm;//cluster_rotation_deg stores the rotation of each clusters. drm is the matrix storing the distance ratio of each cluster pair(the i and j of the matrix indicate the cluster index)

	
	//variables required for histogram 
	float hist_min, hist_max, hist_mode, hist_threshold;// few parameters for obtaining the histogram
	uint hist_count, hist_mode_idx, hist_mode_idx_ll, hist_mode_idx_ul;//few parameters for obtaining the histogram
	vector<float> histogram_dist, histogram_rot;//histogram of all the distance ratios between the cluster pairs and the histogram of rotation between 2 cluster.

	vector<pair<uint,uint> > outlier_ratio_index;// the distance ratios in which one of the 2 clusters is an outlier
	vector<uint> outlier_cluster;// clusters that are outliers

	float rotation_degrees;// average rotation of the camera in degrees

	vector<pair<float,float> > trans_x_y;//stores the translation in x and y for the clusters
	vector<float> hist_trans_x, hist_trans_y;// histogram of translation in x and y
	float x_trans, y_trans;
	uint hist_mode_idx_x_ll, hist_mode_idx_x_ul, hist_mode_idx_y_ll, hist_mode_idx_y_ul;


public:

	// Get Inlier Mask
	// Return number of inliers for GMS matching
	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);
	
	void cluster(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//This function is for the formation of clusters out of the keypoints based on the nearness of the points. The value of the cluster_mask
																										//determines the area where the algorithm will search for keypoints to be pulled into the cluster

	void cluster_matching(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//To give the cluster index/number to the matched keypoints

	void cluster_centroid(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//The function finds the centroid of each cluster of keypoints on both the images. This centroid will be used for further calculation

	void distance_ratio_matrix(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//This function finds the ratio of distances between two clusters in the first image to the that in the second image

	vector<uint> outlier_rejection(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//This function rejects the outlier using the distance ratios in the above function and the correct ratio from the histogram_distance_ratio
																														  //function

	void histogram_distance_ratio(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms);//This function finds the mean distance_ratio by making a histogram of them and finding the maximum repeated ratio within a certain
																														 //range (here hist_threshold). this mean distance_ratio of the clusters is used to remove outliers

	float histogram_rotation_degrees(); //This function is used to find the average rotation in degrees. This would be then used to find the translation in x and y and
	  									//all these values would be published

	void translation(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms_inliers);//this function is used to find the translation of each cluster in x and y in the current frame

	pair<float,float> translation_histogram(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms_inliers);//This function is used to find the average translation of the whole robot in the current frame
	
private:

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const vector<cv::KeyPoint> &kp, const cv::Size &size, vector<cv::Point2f> &npts) {
		const size_t numP = kp.size();
		const int width   = size.width;
		const int height  = size.height;
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const vector<cv::DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	int GetGridIndexLeft(const cv::Point2f &pt, int type) {
		int x = 0, y = 0;

		if (type == 1) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);
		}

		if (type == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);
		}

		if (type == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);
		}

		if (type == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);
		}


		if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height)
		{
			return -1;
		}

		return x + y * mGridSizeLeft.width;
	}

	int GetGridIndexRight(const cv::Point2f &pt) {
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	vector<int> GetNB9(const int idx, const cv::Size& GridSize) {
		vector<int> NB9(9, -1);

		int idx_x = idx % GridSize.width;
		int idx_y = idx / GridSize.width;

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{	
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;

				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;

				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	//
	void InitalizeNiehbors(cv::Mat &neighbor, const cv::Size& GridSize) {
		for (int i = 0; i < neighbor.rows; i++)
		{
			vector<int> NB9 = GetNB9(i, GridSize);
			int *data = neighbor.ptr<int>(i);
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	void SetScale(int Scale) {
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neihbor of right grid 
		mGridNeighborRight = cv::Mat::zeros(mGridNumberRight, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run 
	int run(int RotationType);
};



int gms_matcher::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);
		max_inlier = run(1);
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	if (WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		SetScale(0);
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
			
		}
		return max_inlier;
	}

	return max_inlier;
}

void gms_matcher::AssignMatchPairs(int GridType) {

	for (size_t i = 0; i < mNumberMatches; i++)
	{
		cv::Point2f &lp = mvP1[mvMatches[i].first];
		cv::Point2f &rp = mvP2[mvMatches[i].second];

		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		if (GridType == 1)
		{
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0)	continue;

		mMotionStatistics.at<int>(lgidx, rgidx)++;
		mNumberPointsInPerCellLeft[lgidx]++;
	}
}

void gms_matcher::VerifyCellPairs(int RotationType) {

	const int *CurrentRP = mRotationPatterns[RotationType - 1];

	for (int i = 0; i < mGridNumberLeft; i++)
	{
		if (cv::sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1;
			continue;
		}

		int max_number = 0;
		for (int j = 0; j < mGridNumberRight; j++)
		{
			int *value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				mCellPairs[i] = j;
				max_number = value[j];
			}
		}

		int idx_grid_rt = mCellPairs[i];

		const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt); 

		int score = 0;
		double thresh = 0;
		int numpair = 0;

		for (size_t j = 0; j < 9; j++)
		{
			int ll = NB9_lt[j];
			int rr = NB9_rt[CurrentRP[j] - 1];
			if (ll == -1 || rr == -1)	continue;

			score += mMotionStatistics.at<int>(ll, rr);
			thresh += mNumberPointsInPerCellLeft[ll];
			numpair++;
		}

		thresh = THRESH_FACTOR * sqrt(thresh / numpair);

		if (score < thresh)
			mCellPairs[i] = -2;
	}
}

int gms_matcher::run(int RotationType) {

	mvbInlierMask.assign(mNumberMatches, false); //Initialise the inlier mask

	// Initialize Motion Statisctics
	mMotionStatistics = cv::Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	for (int GridType = 1; GridType <= 4; GridType++) 
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(mGridNumberLeft, -1);
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);
		
		AssignMatchPairs(GridType);
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
			{
				mvbInlierMask[i] = true;
			}
		}
	}
	int num_inlier = cv::sum(mvbInlierMask)[0];
	return num_inlier;
}

void gms_matcher::cluster(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*This function is for the formation of clusters out of the keypoints based on the nearness of the points. The value of the cluster_mask
	  determines the area where the algorithm will search for keypoints to be pulled into the cluster*/
	
	uint16_t temp_count = 0;
	
	for(int i = 0; i<matches_gms.size(); i++)
	{

		float kp1x = kp1[matches_gms[i].queryIdx].pt.x;
		float kp1y = kp1[matches_gms[i].queryIdx].pt.y;

		if((((kp1x + 1 - cluster_mask)*(cluster_index1.cols - cluster_mask - kp1x)) > 0) && (((kp1y + 1 - cluster_mask)*(cluster_index1.rows - cluster_mask - kp1y)) > 0))
		{

			if(cluster_index1.at<uint16_t>(kp1y,kp1x)!=0)
			{	
				temp_idx = cluster_index1.at<uint16_t>(kp1y,kp1x);
				for(int m = -1*cluster_mask; m <= cluster_mask; m++)
				{	
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						if(cluster_index1.at<uint16_t>(kp1y-m, kp1x-n)==0)
							cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = temp_idx;
					}
				}
			}

			else
			{	
				temp_idx = 0;
				dist_idx = 100;
				float d;
				for(int m = -1*cluster_mask; m<=cluster_mask; m++)
				{	
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						d = sqrt(m*m+n*n);
						if((cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) != 0) && (d<dist_idx))
						{	
							temp_idx = cluster_index1.at<uint16_t>(kp1y-m, kp1x-n);
							dist_idx = d;
						}
					}
				}	

				if(temp_idx!=0)
				{
					for(int m = -1*cluster_mask; m<=cluster_mask; m++)
					{
						for(int n = -1*cluster_mask; n<=cluster_mask; n++)
						{
							if(cluster_index1.at<uint16_t>(kp1y-m, kp1x-n)==0)
							cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = temp_idx;
						}
					}
				}	
			}

			if(temp_idx == 0)
			{
				cluster_count++;
				for(int m = -1*cluster_mask; m<=cluster_mask; m++)
				{
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = cluster_count;
					}
				}
			}
		}

		else
		{
			if(cluster_index1.at<uint16_t>(kp1x,kp1y)!=0)
			{	
				temp_idx = cluster_index1.at<uint16_t>(kp1y,kp1x);
				for(int m = -1*cluster_mask; m<=cluster_mask; m++)
				{
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						if(kp1x - n >= 0 && kp1x -n <= cluster_index1.cols && kp1y-m >= 0 && kp1y-m <= cluster_index1.rows)
						{
							if(cluster_index1.at<uint16_t>(kp1y-m, kp1x-n)==0)
								cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = temp_idx;
						}
					}
				}
			}

			else
			{	
				temp_idx = 0;
				dist_idx = 100;
				for(int m = -1*cluster_mask; m<=cluster_mask; m++)
				{
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						uint d = sqrt(m*m+n*n);
						if(kp1x-n >= 0 && kp1x-n <= cluster_index1.cols && kp1y-m >= 0 && kp1y-m <= cluster_index1.rows)	
						{
							if((cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) != 0) && (d<dist_idx))
							{
								temp_idx = cluster_index1.at<uint16_t>(kp1y-m, kp1x-n);
								dist_idx = d;
							}
						}
					}
				}	

				if(temp_idx!=0)
				{
					for(int m = -1*cluster_mask; m<=cluster_mask; m++)
					{
						for(int n = -1*cluster_mask; n<=cluster_mask; n++)
						{
							if(kp1x-n >= 0 && kp1x-n <= cluster_index1.cols && kp1y-m >= 0 && kp1y-m <= cluster_index1.rows)
							{
								if(cluster_index1.at<uint16_t>(kp1y-m, kp1x-n)==0)
									cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = temp_idx;
							}
						}
					}
				}	
			}

			if(temp_idx == 0)
			{
				cluster_count++;
				for(int m = -1*cluster_mask; m<=cluster_mask; m++)
				{
					for(int n = -1*cluster_mask; n<=cluster_mask; n++)
					{
						if(kp1x-n >= 0 && kp1x-n <= cluster_index1.cols && kp1y-m >= 0 && kp1y-m <= cluster_index1.rows)
							cluster_index1.at<uint16_t>(kp1y-m, kp1x-n) = cluster_count;
					}
				}
			}	
		}
	}
}

void gms_matcher::cluster_matching(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*To give the cluster index/number to the matched keypoints*/

	uint16_t temp_count = 0;
	cluster(kp1, kp2, matches_gms);

	for(size_t i=0; i<matches_gms.size(); i++)
	{
		cluster_index2.at<uint16_t>(kp2[matches_gms[i].trainIdx].pt.y, kp2[matches_gms[i].trainIdx].pt.x) = cluster_index1.at<uint16_t>(kp1[matches_gms[i].queryIdx].pt.y, kp1[matches_gms[i].queryIdx].pt.x);
		temp_count = std::max(temp_count, cluster_index2.at<uint16_t>(kp2[matches_gms[i].trainIdx].pt.y, kp2[matches_gms[i].trainIdx].pt.x));
	}
}

void gms_matcher::cluster_centroid(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*The function finds the centroid of each cluster of keypoints on both the images. This centroid will be used for further calculation*/

	cluster_matching(kp1, kp2, matches_gms);
	centroid_query.resize(cluster_count+1);
	centroid_train.resize(cluster_count+1);
	centroid_query[0] = pair<float,float>(0,0);
	centroid_train[0] = pair<float,float>(0,0);
	
	cv::Point2f temp_query, temp_train;
	int temp_n1, temp_n2;
	int kp1x, kp1y, kp2x, kp2y;

	for(size_t i=1; i<=cluster_count; i++)
	{
		temp_query.x = 0;
		temp_query.y = 0;

		temp_train.x = 0;
		temp_train.y = 0;

		temp_n1 = 0;
		temp_n2 = 0;

		for(size_t j=0; j<matches_gms.size(); j++)
		{	
			kp1x = kp1[matches_gms[j].queryIdx].pt.x;
			kp1y = kp1[matches_gms[j].queryIdx].pt.y;

			kp2x = kp2[matches_gms[j].trainIdx].pt.x;
			kp2y = kp2[matches_gms[j].trainIdx].pt.y;

			if(cluster_index1.at<uint16_t>(kp1y, kp1x) == i)
			{
				temp_query.x += kp1x;
				temp_query.y += kp1y;
				temp_n1++;
			}

			if(cluster_index2.at<uint16_t>(kp2y, kp2x) == i)
			{
				temp_train.x += kp2x;
				temp_train.y += kp2y;
				temp_n2++;
			}
		}

		centroid_query[i] = pair<float,float>(1.0f*temp_query.x/temp_n1, 1.0f*temp_query.y/temp_n1);
		centroid_train[i] = pair<float,float>(1.0f*temp_train.x/temp_n2, 1.0f*temp_train.y/temp_n2);
	}
}

void gms_matcher::distance_ratio_matrix(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*This function finds the ratio of distances between two clusters in the first image to the that in the second image*/

	cluster_centroid(kp1, kp2, matches_gms);

	float d1, d2;
	float temp_count = 0;
	uint n = 0;
	
	drm = cv::Mat::zeros(cluster_count + 1, cluster_count + 1, CV_32FC1);
	cluster_rotation_deg = cv::Mat::zeros(cluster_count + 1, cluster_count + 1, CV_32FC1);

	 	
	for(uint i=1; i<=cluster_count; i++)
	{
		for(uint j=1; j<=cluster_count; j++)
		{
			if(i!=j)
			{	

				d1 = sqrt(pow((centroid_query.at(i).first - centroid_query.at(j).first),2) + pow((centroid_query.at(i).second - centroid_query.at(j).second),2));
				d2 = sqrt(pow((centroid_train.at(i).first - centroid_train.at(j).first),2) + pow((centroid_train.at(i).second - centroid_train.at(j).second),2));
				drm.at<float>(i,j) = d1/d2;

				if(i<j)
				{
					d1 = 180/3.141592*atan((centroid_query.at(i).second - centroid_query.at(j).second)/(centroid_query.at(i).first - centroid_query.at(j).first));
					d2 = 180/3.141592*atan((centroid_train.at(i).second - centroid_train.at(j).second)/(centroid_train.at(i).first - centroid_train.at(j).first));
					cluster_rotation_deg.at<float>(i,j) = (d2-d1);
					if(cluster_rotation_deg.at<float>(i,j) > 180)
						cluster_rotation_deg.at<float>(i,j) = -1.0*(360-cluster_rotation_deg.at<float>(i,j));
					if(cluster_rotation_deg.at<float>(i,j) < -180)
						cluster_rotation_deg.at<float>(i,j) = -1.0*(-360-cluster_rotation_deg.at<float>(i,j));
				}
			}
		}
	}	
}

vector<uint> gms_matcher::outlier_rejection(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*This function rejects the outlier using the distance ratios in the above function and the correct ratio from the histogram_distance_ratio
	  function*/

	histogram_distance_ratio(kp1, kp2, matches_gms);

	uint n = 0;
	float temp_count = 0;
	for(uint i=1; i<=cluster_count; i++)
	{
		for(uint j=1; j<=cluster_count; j++)
		{
			if(i!=j)
			{	
				if(drm.at<float>(i,j) <= (hist_min + (hist_mode_idx-1)*hist_threshold) || drm.at<float>(i,j) >= (hist_min + hist_threshold*hist_mode_idx))
				{
					outlier_ratio_index.push_back(pair<uint,uint>(i,j));
				}
				else
				{
					n++;
					temp_count += drm.at<float>(i,j);
				}

			}
		}
	}

	ratio_mean = temp_count/n;
	
	for(uint i=0; i<outlier_ratio_index.size(); i++)
	{	
		temp_count = 0;
		if(i>0 && (outlier_ratio_index.at(i).first != outlier_ratio_index.at(i-1).first))
		{
			for(uint j=0; j<outlier_ratio_index.size(); j++)
			{
				if(outlier_ratio_index.at(j).first == outlier_ratio_index.at(i).first)
					temp_count++;
			}
			if(temp_count>= 0.5*cluster_count || temp_count >=10)
			{

				outlier_cluster.push_back(outlier_ratio_index.at(i).first);
			}
		}

		if(i==0)
		{
			for(uint j=0; j<outlier_ratio_index.size(); j++)
			{
				if(outlier_ratio_index.at(j).first == outlier_ratio_index.at(i).first)
					temp_count++;
			}
			if(temp_count>= 0.6*cluster_count)
			{

				outlier_cluster.push_back(outlier_ratio_index.at(i).first);
			}
		}
	}

	return outlier_cluster;
}

void gms_matcher::histogram_distance_ratio(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms) {
	/*This function finds the mean distance_ratio by making a histogram of them and finding the maximum repeated ratio within a certain
	  range (here hist_threshold). this mean distance_ratio of the clusters is used to remove outliers*/
	distance_ratio_matrix(kp1, kp2, matches_gms);

	for(uint i=1; i<=cluster_count; i++)
	{
		for(uint j=1; j<=cluster_count; j++)
		{
			if(i!=j)
			{
				hist_min = min(hist_min, drm.at<float>(i,j));
				hist_max = max(hist_max, drm.at<float>(i,j));
			}
		}
	}

	hist_count = (hist_max - hist_min)/hist_threshold;

	histogram_dist.push_back(0);

	for(uint m=1; m<=hist_count; m++)
	{	
		uint temp_count = 0;
		for(uint i=1; i<=cluster_count; i++)
		{
			for(uint j=1; j<=cluster_count; j++)
			{
				if(i!=j)
				{
					if(drm.at<float>(i,j) >= (hist_min + (m-1)*hist_threshold) && drm.at<float>(i,j) < (hist_min + hist_threshold*m))
						temp_count++;
				}
			}
		}
		histogram_dist.push_back(temp_count);
		hist_mode = max(histogram_dist.at(m),hist_mode);
		if(hist_mode<=histogram_dist.at(m))
			hist_mode_idx = m;
	}
}

float gms_matcher::histogram_rotation_degrees() {
	/*This function is used to find the average rotation in degrees. This would be then used to find the translation in x and y and
	  all these values would be published*/
	hist_min = 1000.0;
	hist_max = -1000.0;
	hist_mode = 0.0;
	hist_threshold = 2;
	hist_mode_idx = 0;


	for(uint i=1; i<=cluster_count; i++)
	{
		for(uint j=1; j<=cluster_count; j++)
		{
			if(i<j)
			{
				hist_min = min(hist_min, cluster_rotation_deg.at<float>(i,j));
				hist_max = max(hist_max, cluster_rotation_deg.at<float>(i,j));
			}
		}
	}

	hist_count = (hist_max - hist_min)/hist_threshold;

	histogram_rot.push_back(0);

	for(uint m=1; m<=hist_count; m++)
	{	
		uint temp_count = 0;
		for(uint i=1; i<=cluster_count; i++)
		{
			for(uint j=1; j<=cluster_count; j++)
			{
				if(i<j)
				{
					if(cluster_rotation_deg.at<float>(i,j) >= (hist_min + (m-1)*hist_threshold) && cluster_rotation_deg.at<float>(i,j) < (hist_min + hist_threshold*m))
						temp_count++;
				}
			}
		}
		histogram_rot.push_back(temp_count);
		hist_mode = max(histogram_rot.at(m),hist_mode);
		if(hist_mode<=histogram_rot.at(m))
			hist_mode_idx = m;
	}

	
	uint16_t n = 0;
	float count = 0;

	for(uint i=1; i<=cluster_count; i++)
	{
		for(uint j=1; j<=cluster_count; j++)
		{
			if(i<j)
			{	
				if((cluster_rotation_deg.at<float>(i,j)>= (hist_min + (hist_mode_idx-1)*hist_threshold)) && (cluster_rotation_deg.at<float>(i,j)<= (hist_min + hist_mode_idx*hist_threshold)))
				{
					n++;
					count += cluster_rotation_deg.at<float>(i,j);
				}
			}
		}
	}

	rotation_degrees = count/n;

	if(abs(rotation_degrees) >= 0.4)
		return rotation_degrees;
	
	else
	{	
		rotation_degrees = 0;
		return 0;
	}
}

void gms_matcher::translation(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms_inliers) {
	/*this function is used to find the translation of each cluster in x and y in the current frame*/
	float xt, yt, dx1, dy1;
	uint16_t x_cent = cluster_index1.cols/2;
	uint16_t y_cent = cluster_index1.rows/2;

	for(uint16_t i=0; i<matches_gms_inliers.size(); i++)
	{
		dx1 = x_cent - kp1.at(matches_gms_inliers.at(i).queryIdx).pt.x;
		dy1 = kp1.at(matches_gms_inliers.at(i).queryIdx).pt.y - y_cent;
		xt = (x_cent - kp2.at(matches_gms_inliers.at(i).trainIdx).pt.x) - dx1*cos(rotation_degrees*3.141592/180) + dy1*sin(rotation_degrees*3.141592/180);
		yt = (kp2.at(matches_gms_inliers.at(i).trainIdx).pt.y - y_cent) - dy1*cos(rotation_degrees*3.141592/180) - dx1*sin(rotation_degrees*3.141592/180);

		trans_x_y.push_back(pair<float,float>(xt,yt));
	}
}

pair<float,float> gms_matcher::translation_histogram(vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<cv::DMatch> &matches_gms_inliers) {
	/*This function is used to find the average translation of the whole robot in the current frame*/
	translation(kp1, kp2, matches_gms_inliers);

	float hist_min_x = 1000.0, hist_min_y = 1000.0;
	float hist_max_x = -1000.0, hist_max_y = -1000.0;
	float hist_mode_x = 0.0, hist_mode_y = 0.0;
	float hist_mode_idx_x_ll, hist_mode_idx_x_ul;
	hist_threshold = 3;
	uint16_t hist_mode_idx_x = 0, hist_mode_idx_y = 0;

	for(uint i=0; i<matches_gms_inliers.size(); i++)
	{
		hist_min_x = min(hist_min_x, trans_x_y.at(i).first);
		hist_max_x = max(hist_max_x, trans_x_y.at(i).first);
		hist_min_y = min(hist_min_y, trans_x_y.at(i).second);
		hist_max_y = max(hist_max_y, trans_x_y.at(i).second);
	}

	uint16_t hist_count_x = (hist_max_x - hist_min_x)/hist_threshold;
	uint16_t hist_count_y = (hist_max_y - hist_min_y)/hist_threshold;

	hist_trans_x.push_back(0);
	hist_trans_y.push_back(0);

	for(uint m=1; m<=hist_count_x || m<=hist_count_y; m++)
	{	
		uint temp_count_x = 0, temp_count_y = 0;
		for(uint i=0; i<matches_gms_inliers.size(); i++)
		{
			if(m<=hist_count_x)
			{
				if(trans_x_y.at(i).first >= (hist_min_x + (m-1)*hist_threshold) && trans_x_y.at(i).first < (hist_min_x + hist_threshold*m))
				temp_count_x++;
			}

			if(m<=hist_count_y)
			{
				if(trans_x_y.at(i).second >= (hist_min_y + (m-1)*hist_threshold) && trans_x_y.at(i).second < (hist_min_y + hist_threshold*m))
				temp_count_y++;
			}
		}
		hist_trans_x.push_back(temp_count_x);
		hist_trans_y.push_back(temp_count_y);
		
		hist_mode_x = max(hist_trans_x.at(m),hist_mode_x);
		hist_mode_y = max(hist_trans_y.at(m),hist_mode_y);
		
		if(hist_mode_x<=hist_trans_x.at(m))
			hist_mode_idx_x = m;
		if(hist_mode_y<=hist_trans_y.at(m))
			hist_mode_idx_y = m;
	}

	uint16_t nx = 0, ny = 0;
	float count_x = 0, count_y = 0;

	for(uint i=1; i<matches_gms_inliers.size(); i++)
	{
		if(trans_x_y.at(i).first >= (hist_min_x + (hist_mode_idx_x-1)*hist_threshold) && trans_x_y.at(i).first <= (hist_min_x + hist_threshold*hist_mode_idx_x))
		{
			nx++;
			count_x += trans_x_y.at(i).first;
		}
	
		if(trans_x_y.at(i).second >= (hist_min_y + (hist_mode_idx_y-1)*hist_threshold) && trans_x_y.at(i).second <= (hist_min_y + hist_threshold*hist_mode_idx_y))
		{
			ny++;
			count_y += trans_x_y.at(i).second;
		}
	}

	x_trans = count_x/nx;
	y_trans = count_y/ny;

	return pair<float,float>(x_trans, y_trans);
}


inline void imresize(cv::Mat &src, int height) {
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	cv::resize(src, src, cv::Size(width, height));
}
