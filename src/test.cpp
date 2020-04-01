#include "RetinexModel.h"
  
void powMat(cv::Mat& src, cv::Mat& dst, float order)
{
  int nc = src.cols * src.rows;
	dst.create(src.size(), CV_32F);
	float *src_pixel = (float*)src.data;
	float *dst_pixel = (float*)dst.data;
	for (int i = 0; i < nc; i++)
	{
		dst_pixel[i] = pow(src_pixel[i], order);
	}
}
int main()
{
  const cv::Mat I1 = cv::imread("src.bmp",1);//MPI-Sintel/000008_10.png
	if (I1.empty())
	{
		std::cerr << "imread failed." << std::endl;
		return -1;
	}
	cv::Mat src_hsv;
	cv::cvtColor(I1, src_hsv, CV_BGR2HSV);
	std::vector<cv::Mat> V;
	cv::split(src_hsv, V);

	RetinexModel rm;
	cv::Mat reflectance,illumination;
	rm.computer(V[2], reflectance, illumination);
	cv::Mat enhancement;
	cv::Mat enillumination;
	powMat(illumination, enillumination, 0.4);
	cv::multiply(reflectance, enillumination, enhancement);
	cv::normalize(enhancement, enhancement, 0, 255, cv::NORM_MINMAX);
	enhancement.convertTo(enhancement, CV_8U);
	V[2] = enhancement;
	cv::merge(V, enhancement);
	cv::cvtColor(enhancement, enhancement, CV_HSV2BGR);
	cv::imwrite("enhancement.bmp", enhancement);
	cv::normalize(illumination, illumination, 0, 255, cv::NORM_MINMAX);
	illumination.convertTo(illumination, CV_8U);
	V[2] = illumination;
	cv::merge(V, illumination);
	cv::cvtColor(illumination, illumination, CV_HSV2BGR);
	cv::imwrite("illumination.bmp",illumination);

	cv::normalize(reflectance, reflectance, 0, 255, cv::NORM_MINMAX);
	reflectance.convertTo(reflectance, CV_8U);
	V[2] = reflectance;
	cv::merge(V, reflectance);
	cv::cvtColor(reflectance, reflectance, CV_HSV2BGR);
	cv::imwrite("reflectance.bmp", reflectance);
  
  return 0;
}
