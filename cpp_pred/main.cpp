#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "DG.h"

#include <direct.h>

using namespace std;


int main(int argc, char *argv[])
{
	char cbuf[256];

	SSDDetector ssd;
	Classifier classifier;

	vector<cv::String> files;

	sprintf(cbuf, "%s\\*.jpg", argv[1]);
	cv::glob(cv::String(cbuf), files, true);

	cv::String savedir(argv[2]);

	_mkdir(argv[2]);
	
	for (auto f : files)
	{

		//string img_path("E:\\imdb\\dg\\Archive2017.8.16\\2016\\07_23_04_888Ng.jpg");

		cv::Mat raw_img = cv::imread(f);

		vector<cv::Rect> rects;
		ssd.detect(raw_img, rects);
		//cv::resize(raw_img, raw_img, cv::Size(608, 608));
		//vector<cv::Mat> rois;
		vector<int> classes;

		double t = (double)cvGetTickCount();

		for (int i = 0; i < rects.size(); ++i)
		{
			cv::Mat roi = raw_img(rects[i]).clone();
			int c = classifier.classify(roi);
			//int c = 0;
			//rois.push_back(roi);
			classes.push_back(c);
		}

		t = (double)cvGetTickCount() - t;
		printf("CLS time = %gms\n", t / (cvGetTickFrequency() * 1000));

		for (int i = 0; i < rects.size(); ++i)
		{
			cv::Scalar color;
			if (classes[i] == 1)
				color = cv::Scalar(0, 0, 255);
			else
				color = cv::Scalar(0, 255, 0);

			cv::rectangle(raw_img, rects[i], color, 8);
		}

		//if (raw_img.cols > 1000 || raw_img.rows > 1000)
		//{
		//	cv::resize(raw_img, raw_img, cv::Size(), 0.3, 0.3);
		//	cv::imshow("w", raw_img);
		//	cv::waitKey();
		//}
		//cv::String fn;
		//f.substr(f.find_last_of("\\"))

		cv::imwrite(savedir + f.substr(f.find_last_of("\\")), raw_img);
	}
	return 0;
}