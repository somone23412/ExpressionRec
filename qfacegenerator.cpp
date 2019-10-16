#include "stdafx.h"

//qt
#include "qfacegenerator.h"
#include <QDir>
#include <QFile>

//std
#include <vector>
#include <string>
#include <cstdlib>

//for caffe layer registe
#include "caffexxx.h"

//FeatureMapGenerator
#include "FeatureMapGenerator.h"

//ExpressionRec
#include "MTCNN.h"
#include "ExpressionModel.h"
#include "FaceUtils.h"

//Time test
#include <time.h>

//File io
#include <io.h>

cv::Mat calcDiff(cv::Mat inputImg, cv::Mat biResizeImg){
	cv::Mat diffImg = cv::Mat::zeros(128, 128, CV_8UC3);
	for (int h = 0; h < 128; h++) {
		for (int w = 0; w < 128; w++) {
			diffImg.at<cv::Vec3b>(h, w)[0] = inputImg.at<cv::Vec3b>(h, w)[0] - biResizeImg.at<cv::Vec3b>(h, w)[0];
			diffImg.at<cv::Vec3b>(h, w)[1] = inputImg.at<cv::Vec3b>(h, w)[1] - biResizeImg.at<cv::Vec3b>(h, w)[1];
			diffImg.at<cv::Vec3b>(h, w)[2] = inputImg.at<cv::Vec3b>(h, w)[2] - biResizeImg.at<cv::Vec3b>(h, w)[2];
		}
	}
	return diffImg;
}

cv::Mat calcAbsDiff(cv::Mat inputImg, cv::Mat biResizeImg){
	cv::Mat absDiffImg = cv::Mat::zeros(128, 128, CV_8UC3);
	for (int h = 0; h < 128; h++) {
		for (int w = 0; w < 128; w++) {
			absDiffImg.at<cv::Vec3b>(h, w)[0] = abs(inputImg.at<cv::Vec3b>(h, w)[0] - biResizeImg.at<cv::Vec3b>(h, w)[0]);
			absDiffImg.at<cv::Vec3b>(h, w)[1] = abs(inputImg.at<cv::Vec3b>(h, w)[1] - biResizeImg.at<cv::Vec3b>(h, w)[1]);
			absDiffImg.at<cv::Vec3b>(h, w)[2] = abs(inputImg.at<cv::Vec3b>(h, w)[2] - biResizeImg.at<cv::Vec3b>(h, w)[2]);
		}
	}
	return absDiffImg;
}

void getFiles(std::string path, std::vector<std::string>& files, std::vector<std::string>& filesname)
{
	//文件句柄  
	//long hFile = 0;  //win7
	intptr_t hFile = 0;   //win10
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
		// "\\*"是指读取文件夹下的所有类型的文件，若想读取特定类型的文件，以png为例，则用“\\*.png”
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("/").append(fileinfo.name), files, filesname);
			}
			else
			{
				files.push_back(path + "/" + fileinfo.name);
				filesname.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

QFaceGenerator::QFaceGenerator(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	//choose model, 0 = VGGNet, 1 = mobileNet
	int choose_model = 2;
	std::string model_iters = "";

	//img to test
	std::string test_img_path = "faceimg/expression_test/";
	std::string test_result_path = "faceimg/expression_test_images_result/";
	std::vector<std::string> test_img_list = {
		"2019-09-09-154916.jpg",
		"2019-09-09-154928.jpg",
		"2019-09-09-154940.jpg",
		"2019-09-09-155040.jpg",
		"2019-09-09-155049.jpg",
		"2019-09-09-155106.jpg",
		"2019-09-09-155121.jpg",
		"2019-09-09-155212.jpg",
		//"test_0066.jpg",
		//"test_0093.jpg",
		//"test_0125.jpg",
		//"test_0160.jpg",
		//"test_0195.jpg",
		//"test_0201.jpg",
		//"test_0209.jpg",
		//"test_0233.jpg",
		//"test_0283.jpg",
		//"test_0294.jpg",
		//"test_0538.jpg",
	};

	std::vector<std::vector<std::string>> filesname;
	for (int i = 0; i < 7; i++) {
		std::vector <std::string> f;
		std::vector <std::string> fn;
		getFiles(test_img_path + std::to_string(i), f, fn);
		filesname.push_back(fn);
	}

	std::cout << "\nfiles" << std::endl;
	for (int i = 0; i < 7; i++) {
		std::cout << i << "#" << filesname[i].size() << std::endl;
	}

	//mtcnn config
	std::vector<std::string> mtcnn_model_file = {
		"model/det1.prototxt",
		"model/det2.prototxt",
		"model/det3.prototxt"
	};
	std::vector<std::string> mtcnn_trained_file = {
		"model/det1.caffemodel",
		"model/det2.caffemodel",
		"model/det3.caffemodel"
	};
	std::shared_ptr<MTCNN> mtcnn = std::make_shared<MTCNN>(mtcnn_model_file, mtcnn_trained_file);

	//expression model config
	std::string target_layer = "prob";
	std::shared_ptr<ExpressionModel> expressionModel;
	float scale = 0;
	std::vector<float> meanValue;
	if (choose_model == 0) {
		std::cout << "vggnet" << std::endl;
		scale = 1.0;
		meanValue = { 129.1863f, 104.7624f, 93.5940f };
		expressionModel = std::make_shared<ExpressionModel>("model/RAF.prototxt", "model/RAF__iter_7500.caffemodel");
	}else if (choose_model == 1) {
		std::cout << "mobilenet" << std::endl;
		scale = 0.017;
		meanValue = { 103.94f, 116.78f, 123.68f };
		expressionModel = std::make_shared<ExpressionModel>("model/depthwise_mobileNet.prototxt", "model/mobileNet__iter_8000.caffemodel");
	}else if (choose_model == 2) {
		std::cout << "lightcnn" << std::endl;
		scale = 0.0078125f / 2;
		meanValue = { 0.0f, 0.0f, 0.0f };
		expressionModel = std::make_shared<ExpressionModel>("model/lightCNN_deploy.prototxt", "model/lightCNN__iter_6000.caffemodel");
	}

	//set Model params
	expressionModel->setScale(scale);
	expressionModel->setMeanValue(meanValue);
	std::vector<std::string> layerNames = {
		target_layer,
	};

	//vector<float> meanValue = { (float)131.0912, (float)103.8827, (float)91.4953 };
	//expressionModel->setMeanValue(meanValue);
	//expressionModel->setScale(1.0);
	//cv::Mat inputImg = cv::imread("faceimg/183089.png");
	//cv::imshow("inputImg", inputImg);

	//application interface
	cv::VideoCapture expressionCapture;
	int frame_internal = 30;
	expressionCapture.open(0);
	if (!expressionCapture.isOpened()) {
		std::cout << "[error] qfacegenerator.cpp: line 81, open capture failed." << std::endl;
	}
	cv::Mat frame;
	while (true) {
		expressionCapture >> frame;
	//	//cv::resize(frame, frame, cv::Size(frame.cols * 2, frame.rows * 2));
	//for (int exp = 0; exp < 7; exp++)
		//for (std::string img_name : filesname[exp]) {
			//frame = cv::imread(test_img_path + std::to_string(exp) + "/" + img_name);
	//for (std::string img_name : test_img_list) {
	//		frame = cv::imread(test_img_path + "/" + img_name);
			if (choose_model == 2) {
				cvtColor(frame, frame, CV_BGR2GRAY);
			}
			if (!frame.empty()) {
				//forward model
				vector<cv::Rect> rectangles_p;
				vector<float> confidences_p;
				std::vector<std::vector<cv::Point>> landmark;
				std::vector<float> feature;
			
				//detect face
				clock_t m_start = clock();
				mtcnn->minSize_ = min(frame.cols / 8, frame.rows / 8);
				mtcnn->detection(frame, rectangles_p, confidences_p, landmark);
				clock_t m_end = clock();
				std::cout << "MTCNN：" << m_end - m_start << "ms" << std::endl;
				int probe_face_num = rectangles_p.size();
			
				if (probe_face_num > 0) {
					int max_bbx_idx = 0; //face detected

					//select max bounding box face
					for (int i = 0; i < probe_face_num; i ++)
						if (rectangles_p[i].width * rectangles_p[i].height > rectangles_p[max_bbx_idx].width * rectangles_p[max_bbx_idx].height) {
							max_bbx_idx = i;
						}

					//drwa bbx and landmark
					for (int j = 0; j < 5; j++){
						cv::circle(frame, landmark[max_bbx_idx][j], 2, CV_RGB(0, 255, 0));
					}

					cv::rectangle(
						frame, 
						cvPoint(rectangles_p[max_bbx_idx].x, rectangles_p[max_bbx_idx].y), 
						cvPoint(rectangles_p[max_bbx_idx].x + rectangles_p[max_bbx_idx].width, rectangles_p[max_bbx_idx].y + rectangles_p[max_bbx_idx].height),
						CV_RGB(255, 0, 0),
						1
					);

					//aligncrop, feature extract, and get target feature
					cv::Mat crop_img = fu::AlignCrop(frame, landmark[max_bbx_idx]);
					//center crop resize to 256 and crop 224
					cv::resize(crop_img, crop_img, cv::Size(144, 144));
					crop_img = crop_img(cv::Rect(8, 8, 128, 128));
					clock_t e_start = clock();
					expressionModel->Forward(crop_img, layerNames);
					clock_t e_end = clock();
					std::cout << "Expression Model：" << e_end - e_start << "ms" << std::endl;
					std::unordered_map<std::string, vector<float>> features = expressionModel->getFeatures();
					feature = features[target_layer];

					//cout layers shape
					//for (std::pair<std::string, vector<float>> p : features){
					//	std::cout << p.first << " " << p.second.size() << std::endl;
					//}

					//cout target feature
					//for (float f : feature){
					//	std::cout << f << " ";
					//}
					//std::cout << std::endl;

					//translate feature to Expression
					std::string expression = expressionModel->getExpression(feature);
					cv::putText(
						frame,
						expression,
						cvPoint(rectangles_p[max_bbx_idx].x, rectangles_p[max_bbx_idx].y),
						cv::FONT_HERSHEY_COMPLEX,
						2, //font size
						CV_RGB(0, 255, 0),
						2, 8, 0
					);
				}

				//draw Expression hist base line
				int drawMargin = frame.cols * 0.025;
				int drawSize = frame.cols * 0.95;
				int bottomEdge = frame.rows - 3 * drawMargin > 0 ? frame.rows - 3 * drawMargin : 0;
				float histFontSize = frame.cols / (float)500;
				int leftEdge = drawMargin < frame.cols ? drawMargin : frame.cols;
				int rightEdge = drawSize + drawMargin < frame.cols ? drawSize + drawMargin : frame.cols;
				//std::cout << " bottomEdge=" << bottomEdge << " leftEdge=" << leftEdge << " rightEdge=" << rightEdge << std::endl;
				cv::line(frame, Point(leftEdge, bottomEdge), Point(rightEdge, bottomEdge), CV_RGB(10, 230, 10), 1);
			
				int expressionNum = expressionModel->getExpressionNum();
				//if detected face, draw hist
				if (feature.size() == expressionNum) {
					int max_idx = 0;
					//get max feature idx
					for (int i = 0; i < expressionNum; i++) {
						if (feature[i] > feature[max_idx]) {
							max_idx = i;
						}
					}
					//blankLevel = blank width / retectanle width 
					int blankLevel = 3;
					//e = minum draw width
					int e = drawSize / ((blankLevel + 1) * expressionNum);
					for (int i = 0; i < expressionNum; i++) {
						cv::Scalar drawColor = CV_RGB(10, 245, 10); //green
						if (i == max_idx) {
							drawColor = CV_RGB(245, 10, 10); // max red
						}
						cv::rectangle(
							frame,
							cvPoint((blankLevel + 1) * e * i + leftEdge, bottomEdge - feature[i] * 100),
							cvPoint((blankLevel + 1) * e * i + e + leftEdge, bottomEdge),
							drawColor,
							-1 //fill
						);
						cv::putText(
							frame,
							expressionModel->getExpressionMap()[i],
							cvPoint((blankLevel + 1) * e * i + leftEdge, bottomEdge + e),
							cv::FONT_HERSHEY_COMPLEX,
							0.6, //font scale
							CV_RGB(10, 245, 10),
							2, 8, 0
						);
					}
					//string result = (max_idx == exp) ? "true" : "false";
					//cv::imwrite(test_result_path + std::to_string(exp) + "/" + result + "_" + img_name, frame);
					//cv::imwrite(test_result_path + "/lightcnn6000_" + img_name, frame);
					//cout << test_result_path + std::to_string(exp) + "/" + result + "_" + img_name << endl;
				}else {
					//failed to detect faces
					//cv::imwrite(test_result_path + std::to_string(exp) + "/df_" + img_name, frame);
				}

				//show frame
				cv::imshow("expressionCapture", frame);

			}else {
				std::cout << "[error] qfacegenerator.cpp: line 91, empty frame." << std::endl;
			}

			if (waitKey(frame_internal) > 0) {
				break;
			}
		}

	////test bi-resize mothod
	//cv::Mat inputImg, biResizeImg1, biResizeImg2, biResizeImg3;
	//cv::Mat I1, I2, I3, I4;
	//cv::Mat diffImg, absDiffImg;

	//inputImg = cv::imread("faceimg/183089.png");
	//cv::imshow("inputImg", inputImg);

	//cv::resize(inputImg, inputImg, cv::Size(128, 128));

	//cv::resize(inputImg, I1, cv::Size(20, 20));
	//cv::resize(inputImg, I2, cv::Size(19, 19));
	//cv::resize(inputImg, I3, cv::Size(18, 18));

	//cv::resize(I1, biResizeImg1, cv::Size(128, 128));
	//cv::resize(I2, biResizeImg2, cv::Size(128, 128));
	//cv::resize(I3, biResizeImg3, cv::Size(128, 128));

	//cv::Mat diffImg1 = calcDiff(biResizeImg1, biResizeImg2);
	//cv::Mat diffImg2 = calcDiff(biResizeImg2, biResizeImg3);

	//diffImg = calcDiff(diffImg1, diffImg2);

	//absDiffImg = calcAbsDiff(diffImg1, diffImg2);
	//
	//cv::imshow("diffImg1", diffImg1);
	//cv::imshow("diffImg2", diffImg2);

	//cv::imshow("diffImg", diffImg);
	//cv::imshow("absDiffImg", absDiffImg);



	////set up featureMapGenerator
	//FeatureMapGenerator *featureMapGenerator = new FeatureMapGenerator("model/fnm.prototxt", "model/fnm.caffemodel");
	////vector<float> meanValue = { (float)0, (float)0, (float)0};
	//vector<float> meanValue = { (float)131.0912, (float)103.8827, (float)91.4953 };
	//featureMapGenerator->setMeanValue(meanValue);
	//featureMapGenerator->setScale(1.0);
	//std::vector<std::string> layerNames = {
	//	"data",
	//	"conv1_7x7_s2_bn",
	//	"conv2_1_1x1_reduce",
	//	"conv2_2_1x1_reduce",
	//	"conv2_3_1x1_reduce",
	//	"conv3_1_1x1_reduce",
	//	"conv3_2_1x1_reduce",
	//	"decoder/res3/conv2",
	//	"decoder/res4/conv2",
	//	"decoder/res5/conv2",
	//	"decoder/res6/conv2",
	//	"decoder/cw_conv/pw_conv",
	//};
	//
	////generate img via cv::Mat
	//cv::Mat faceImg, inputImg, resizeImg, genImg;
	//faceImg = cv::imread("faceimg/2019_05_08_161026837.jpg");
	////cv::resize(faceImg, resizeImg, cv::Size(128, 128));

	//////mask
	////for (int h = 0; h < 128; h++) {
	////	for (int w = 0; w < 128; w++) {
	////		if (h > 35 && h < 70 && w > 35 && w < 70 && abs(h - w) < 15) {
	////			resizeImg.at<cv::Vec3b>(h, w)[0] = rand() % 255;
	////			resizeImg.at<cv::Vec3b>(h, w)[1] = rand() % 255;
	////			resizeImg.at<cv::Vec3b>(h, w)[2] = rand() % 255;
	////		}
	////	}
	////}

	//inputImg = featureMapGenerator->generateFeatureMaps(faceImg, layerNames)["data"][3];
	//
	//cv::imshow("inputImg", faceImg);

	//genImg = featureMapGenerator->getFeatureMaps()["decoder/cw_conv/pw_conv"][3];
	//cv::imshow("genImg", genImg);


	//QDir dir;
	//QString curDir = dir.currentPath();

	////all img
	//auto genImgs = featureMapGenerator->getFeatureMaps();
	//int i = 0;
	//for (auto name : layerNames) {
	//	QString path = QString::fromStdString("featureMaps/" + std::to_string(i) + name + "/");
	//	//std::cout << path.toStdString() << endl;
	//	if (!dir.exists(path)){
	//		dir.mkpath(path);
	//	}
	//	int j = 0;
	//	for (auto img : genImgs[name]) {
	//		cv::imwrite(path.toStdString() + "gen " + "_" + std::to_string(j) + ".jpg", img);
	//		j++;
	//	}
	//	i++;
	//}


}

QFaceGenerator::~QFaceGenerator()
{

}
