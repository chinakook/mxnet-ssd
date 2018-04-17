#pragma once
#include <vector>
#include <mxnet/my_predict_api.h>
#include <opencv2/opencv.hpp>

using std::vector;

class SSDDetector
{
public:
	SSDDetector();

	~SSDDetector();

	int detect(const cv::Mat imgin, vector<cv::Rect>& rects);

private:
	int data_shape_;
	float mean_r_;
	float mean_g_;
	float mean_b_;
	float std_scale_;
	PredictorHandle predictor_;
};

class Classifier
{
public:
	Classifier();

	~Classifier();

	int classify(const cv::Mat imgin);

private:
	int data_shape_;
	float mean_r_;
	float mean_g_;
	float mean_b_;
	float std_scale_;
	PredictorHandle predictor_;
};