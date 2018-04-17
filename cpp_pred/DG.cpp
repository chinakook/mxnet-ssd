#include "DG.h"
#include <string>

using namespace std;

int loadMXSymbol(string json_file, string& json_buffer)
{
	ifstream json_handle(json_file, std::ios::ate);
	json_buffer.reserve(json_handle.tellg());
	json_handle.seekg(0, std::ios::beg);
	json_buffer.assign(std::istreambuf_iterator<char>(json_handle), std::istreambuf_iterator<char>());
	if (json_buffer.size() < 1)
	{
		return -1;
	}

	return 0;
}

int loadMXModel(string model_file, vector<char>& param_buffer)
{
	ifstream param_file(model_file, std::ios::binary | std::ios::ate);
	if (!param_file.is_open())
	{
		return -1;
	}

	streamsize size = param_file.tellg();
	param_file.seekg(0, std::ios::beg);

	param_buffer.resize(size);
	if (param_file.read(param_buffer.data(), size))
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

void wrapCVMat(const cv::Mat &img, vector<float>& img_buffer, float mean_r, float mean_g, float mean_b, float std_scale)
{
	cv::Mat img_f;
	img.convertTo(img_f, CV_32F);

	img_buffer.resize(3 * img_f.rows*img_f.cols);
	cv::Mat rgb[3];
	rgb[0] = cv::Mat(img_f.rows, img_f.cols, CV_32F, &img_buffer[2 * img_f.rows*img_f.cols]);
	rgb[1] = cv::Mat(img_f.rows, img_f.cols, CV_32F, &img_buffer[1 * img_f.rows*img_f.cols]);
	rgb[2] = cv::Mat(img_f.rows, img_f.cols, CV_32F, &img_buffer[0]);
	cv::split(img_f, rgb);

	rgb[0] = std_scale*(rgb[0] - mean_r);
	rgb[1] = std_scale*(rgb[1] - mean_g);
	rgb[2] = std_scale*(rgb[2] - mean_b);
}

SSDDetector::SSDDetector()
{
	string model_file = "model/deploy_ssd_mobilenet_little_608-0060.params";
	string json_file = "model/deploy_ssd_mobilenet_little_608-symbol.json";

	string json_buffer;
	if (loadMXSymbol(json_file, json_buffer) != 0)
		return;

	vector<char> param_buffer;
	if (loadMXModel(model_file, param_buffer) != 0)
		return;

	data_shape_ = 608;
	mean_r_ = 123.f;
	mean_g_ = 117.f;
	mean_b_ = 104.f;
	std_scale_ = 0.017f;

	int device_type = 1;
	int device_id = 0;
	const char *input_keys[1];
	input_keys[0] = "data";
	const mx_uint input_shape_indptr[] = { 0, 4 };
	const mx_uint input_shape_data[] = { 1,3,static_cast<mx_uint>(data_shape_), static_cast<mx_uint>(data_shape_) };

	predictor_ = NULL;

	int err = MYPredCreate(json_buffer.c_str(), param_buffer.data(), static_cast<int>(param_buffer.size()),
		device_type, device_id, 1, input_keys, input_shape_indptr, input_shape_data, &predictor_);

	if (err)
	{
		return;
	}
}

SSDDetector::~SSDDetector()
{
	MYPredFree(predictor_);
}

int SSDDetector::detect(const cv::Mat imgin, vector<cv::Rect>& rects)
{
	int w = imgin.cols;
	int h = imgin.rows;
	double ascpect_x =  w / (double)data_shape_ ;
	double ascpect_y = h / (double)data_shape_;
	cv::Mat img;
	cv::resize(imgin, img, cv::Size(data_shape_, data_shape_));

	vector<float> img_buffer;
	wrapCVMat(img, img_buffer, mean_r_, mean_g_, mean_b_, std_scale_);

	const mx_uint input_shape_data[] = { 1,3,static_cast<mx_uint>(data_shape_), static_cast<mx_uint>(data_shape_) };

	MYPredSetInput(predictor_, "data", &img_buffer[0], static_cast<mx_uint>(3 * img.rows*img.cols));

	double t = (double)cvGetTickCount();

	MYPredForward(predictor_);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;
	MYPredGetOutputShape(predictor_, 0, &shape, &shape_len);


	mx_uint tt_size = 1;
	for (mx_uint i = 0; i < shape_len; ++i)
	{
		tt_size *= shape[i];
	}
	if (tt_size % 6 != 0)
	{
		printf("invalid inference shape.\n");
		return -1;
	}
	vector<float> outputs(tt_size);

	MYPredGetOutput(predictor_, 0, outputs.data(), tt_size);

	t = (double)cvGetTickCount() - t;
	printf("DET time = %gms\n", t / (cvGetTickFrequency() * 1000));

	vector<vector<float> > final_dets;

	for (mx_uint i = 0; i < shape[1]; ++i)
	{
		vector<float> det;
		float cls_id = outputs[shape[0] * i * 6 + 0];
		if (cls_id >= 0)
		{
			float score = outputs[shape[0] * i * 6 + 1];
			if (score > 0.6)
			{
				float xmin = outputs[shape[0] * i * 6 + 2] * data_shape_ * ascpect_x;
				float ymin = outputs[shape[0] * i * 6 + 3] * data_shape_ * ascpect_y;
				float xmax = outputs[shape[0] * i * 6 + 4] * data_shape_ * ascpect_x;
				float ymax = outputs[shape[0] * i * 6 + 5] * data_shape_ * ascpect_y;

				det.push_back(xmin);
				det.push_back(ymin);
				det.push_back(xmax);
				det.push_back(ymax);
				det.push_back(score);
				det.push_back(cls_id);

				final_dets.push_back(det);
			}
		}
	}

	rects.resize(0);
	for (int i = 0; i < final_dets.size(); ++i)
	{
		int x0 = std::max(0, int(final_dets[i][0]));
		int y0 = std::max(0, int(final_dets[i][1]));
		int x1 = std::min(w - 1, int(final_dets[i][2]));
		int y1 = std::min(h - 1, int(final_dets[i][3]));

		if (x0 >= x1 || y0 >= y1)
			continue;

		rects.push_back(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
	}

	return rects.size();
}

Classifier::Classifier()
{
	string model_file = "model/dggluon-0000.params";
	string json_file = "model/dggluon-symbol.json";

	string json_buffer;
	if (loadMXSymbol(json_file, json_buffer) != 0)
		return;

	vector<char> param_buffer;
	if (loadMXModel(model_file, param_buffer) != 0)
		return;

	data_shape_ = 224;
	mean_r_ = 123.f;
	mean_g_ = 117.f;
	mean_b_ = 104.f;
	std_scale_ = 0.017f;

	int device_type = 1;
	int device_id = 0;
	const char *input_keys[1];
	input_keys[0] = "data";
	const char *out_keys[1];
	out_keys[0] = "mobilenetv20_output_pred_fwd";
	const mx_uint input_shape_indptr[] = { 0, 4 };
	const mx_uint input_shape_data[] = { 1,3,static_cast<mx_uint>(data_shape_), static_cast<mx_uint>(data_shape_) };

	predictor_ = NULL;

	int err = MYPredCreatePartialOut(json_buffer.c_str(), param_buffer.data(), static_cast<int>(param_buffer.size()),
		device_type, device_id, 1, input_keys, input_shape_indptr, input_shape_data, 1, out_keys, &predictor_);

	if (err)
	{
		return;
	}
}
Classifier::~Classifier()
{
	MYPredFree(predictor_);
}

int Classifier::classify(const cv::Mat imgin)
{
	cv::Mat img;
	cv::resize(imgin, img, cv::Size(data_shape_, data_shape_));
	//cv::imshow("xx", img);
	//cv::waitKey();
	vector<float> img_buffer;
	wrapCVMat(img, img_buffer, mean_r_, mean_g_, mean_b_, std_scale_);

	MYPredSetInput(predictor_, "data", &img_buffer[0], static_cast<mx_uint>(3 * img.rows*img.cols));

	MYPredForward(predictor_);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;
	MYPredGetOutputShape(predictor_, 0, &shape, &shape_len);

	mx_uint tt_size = 1;
	for (mx_uint i = 0; i < shape_len; ++i)
	{
		tt_size *= shape[i];
	}

	vector<float> outputs(tt_size);
	//printf("%d\n",outputs.size());

	MYPredGetOutput(predictor_, 0, outputs.data(), tt_size);
	//printf("%f %f\n", outputs[0], outputs[1]);
	return outputs[0] < outputs[1];
}