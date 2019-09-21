#include "stdafx.h"
#include "ExpressionModel.h"

#define EMDEBUG
//#define SHOW_FEATURE

std::vector<std::string> expressionMap = {
	"Surprise",
	"Fear",
	"Disgust",
	"Happiness",
	"Sadness",
	"Anger",
	"Neutral",
};

ExpressionModel::ExpressionModel(std::string modelFile, std::string trainedFile)
	:scale(1.0), meanValue({ 129.1863f, 104.7624f, 93.5940f }), expressionNum(7) {
	this->net = new caffe::Net<float>(modelFile, caffe::TEST);
	this->net->CopyTrainedLayersFrom(trainedFile);

	this->input_layer = this->net->input_blobs()[0];
}


ExpressionModel::~ExpressionModel() {
	delete this->net;
}


std::unordered_map<std::string, std::vector<float>> ExpressionModel::Forward(cv::Mat img, std::vector<std::string> &layerNames) {
	//clean featureMaps
	this->features.clear();

	//set Params
	caffe::TransformationParameter tp;
	tp.set_scale(this->scale);
	tp.add_mean_value(this->meanValue[0]);
	tp.add_mean_value(this->meanValue[1]);
	tp.add_mean_value(this->meanValue[2]);

	//trans Mat to Caffe-Need-Type (#define USE_OPENCV)
	caffe::DataTransformer<float> dt(tp, caffe::Phase::TEST);
	cv::Mat tImg;
	//resize to input layer size
	cv::resize(img, tImg, cv::Size(input_layer->width(), input_layer->height()));
	dt.Transform(tImg, input_layer);

	//feature extract
	this->net->ForwardFrom(0);

	//trans each layer's feature to Mat
	for (int i = 0; i < layerNames.size(); i++) {
		caffe::Blob<float> *tmpLayer = net->blob_by_name(layerNames[i]).get();
		const float *begin = tmpLayer->cpu_data();
		const int length = tmpLayer->channels() * tmpLayer->width() * tmpLayer->height();
		float *feat = new float[length];

#ifdef FGDEBUG
		std::cout << "[layer" << " : " << layerNames[i] << "]" << std::endl;
		std::cout << "channels" << " : " << tmpLayer->channels();
		std::cout << " | width" << " : " << tmpLayer->width();
		std::cout << " | height" << " : " << tmpLayer->height();
		std::cout << " | data length" << " : " << length << std::endl << std::endl;
#endif

		const float *end = tmpLayer->cpu_data() + length;
		::memcpy(feat, begin, sizeof(float) * length);

#ifdef SHOW_FEATURE
		//print feature
		for (int i = 0; i < length; i++) {
			std::cout << i << ":" << feat[i] << " ";
			if (0 == (i + 1) % 5 || length - 1 == i) {
				std::cout << std::endl;
			}
		}
#endif

		//save result
		std::vector<float> tmp_feature;
		for (int i = 0; i < length; i++) {
			tmp_feature.push_back(feat[i]);
		}
		this->features.insert({ layerNames[i], tmp_feature });

		delete feat;
	}
	return this->features;
}


std::unordered_map<std::string, std::vector<float>> ExpressionModel::Forward(std::string imgPath, std::vector<std::string> &layerNames){
	cv::Mat img = cv::imread(imgPath);
	return this->Forward(img, layerNames);
}




void ExpressionModel::setMeanValue(std::vector<float> &meanValue) {
	this->meanValue = meanValue;
}


void ExpressionModel::setScale(float scale) {
	this->scale = scale;
}


std::vector<float> ExpressionModel::getMeanValue() const {
	return this->meanValue;
}


float ExpressionModel::getScale() const {
	return this->scale;
}

std::unordered_map<std::string, std::vector<float>> ExpressionModel::getFeatures() const {
	return this->features;
}

std::string ExpressionModel::getExpression(std::vector<float> &feature){
	if (feature.size() == expressionNum) {
		int max_idx = 0;
		for (int i = 0; i < expressionNum; i++) {
			if (feature[i] > feature[max_idx]) {
				max_idx = i;
			}
		}
		return expressionMap[max_idx];
	}
	return "Null";
}

int ExpressionModel::getExpressionNum() const {
	return expressionNum;
}

std::vector<std::string> ExpressionModel::getExpressionMap() const {
	return expressionMap;
}