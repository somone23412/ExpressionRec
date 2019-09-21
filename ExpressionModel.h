#pragma once

#include <opencv2\opencv.hpp>

#include <caffe\caffe.hpp>
#include <caffe\proto\caffe.pb.h>
#include <caffe\data_transformer.hpp>

#include <vector>
#include <string>
#include <unordered_map>

class ExpressionModel {
public:
	ExpressionModel(std::string modelFile, std::string trainedFile);
	~ExpressionModel();

	void setMeanValue(std::vector<float> &meanValue);
	void setScale(float scale);

	std::vector<float> getMeanValue() const;
	float getScale() const;
	int getExpressionNum() const;
	std::vector<std::string> getExpressionMap() const;
	std::unordered_map<std::string, std::vector<float>> getFeatures() const;
	std::string getExpression(std::vector<float> &feature);
	std::unordered_map<std::string, std::vector<float>> Forward(cv::Mat img, std::vector<std::string> &layerNames);
	std::unordered_map<std::string, std::vector<float>> Forward(std::string imgPath, std::vector<std::string> &layerNames);

private:
	std::vector<cv::Mat> transToMat(float* feat, caffe::Blob<float> *layer);

private:
	caffe::Net<float> *net;
	caffe::Blob<float> *input_layer;

	float scale;
	int expressionNum;
	std::vector<float> meanValue;

	std::vector<std::string> layerNames;
	std::unordered_map<std::string, std::vector<float>> features;
};