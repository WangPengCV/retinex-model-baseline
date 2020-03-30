/*
*Copyright (c) 2020/2/24 NUAA
*All rights reserved
*
*name:RetinexModel.cpp
*Abstract:Retinex decomposition baseline
*Version 1.0
*Writer: Peng Wang
*Complete: 2020/2/28 at home
*modify:
*/
#pragma once
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <opencv2/opencv.hpp>

class RetinexModel
{
public:
	struct Parameters
	{
		float lamda_r;
		float lamda_i;

		float gamma_s;
		float gamma_t;

		int K;

		float epsilon;
		Parameters()
		{
			lamda_r = 0.0001;
			lamda_i = 0.002;

			gamma_s = 1.5;
			gamma_t = 0.5;

			K = 20;

			epsilon = 0.01;



		}
	};
	RetinexModel(const Parameters& param = Parameters());

	void computer(const cv::Mat& src, cv::Mat& reflectance, cv::Mat&  illumination);

	void diff(float*& src, float*& dx, float*& dy);

	void weightingmatrix(cv::Mat& dx, cv::Mat& dy, cv::Mat& Sx, cv::Mat& Sy,float gamma);

	void solveclosesolution(cv::Mat& img, cv::Mat& Pori_x,cv::Mat& pori_y,cv::Mat& fiximg,cv::Mat& output,float lamda);

	~RetinexModel();

private:
	Parameters _param;
	int _img_rows;
	int _img_cols;
};

