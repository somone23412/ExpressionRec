#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>


namespace fu{
	struct face_landmark
	{
		float x[5];
		float y[5];
	};

	cv::Mat AlignCrop(cv::Mat src_img, std::vector<cv::Point> &landmark);
	cv::Mat getSimilarityTransformMatrix(float src[5][2]);
	float FeatureCompare(
		std::vector<float> &feat1,
		std::vector<float> &feat2,
		int dims
		);

};