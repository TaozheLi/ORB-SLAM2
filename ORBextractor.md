# ORB feature extractor module
This module is related to the feature extraction.

## Important functions is ORBextractor are as follows:

'''
void ComputePyramid(cv::Mat image); //related to make Pyramid Image;
std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX, /
               const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level); // related to select distributed keypoints
'''
