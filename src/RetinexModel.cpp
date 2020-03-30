#include "RetinexModel.h"
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

RetinexModel::RetinexModel(const Parameters& param )
{

}
RetinexModel::~RetinexModel()
{

}

static float FnormMatrix(cv::Mat src)
{
	int nc = src.cols * src.rows;
	float *src_pixel = (float*)src.data;
	float value = 0;
	for (int i = 0; i < nc; i++)
	{
		value += src_pixel[i] * src_pixel[i];
	}

	return sqrt(value);

}

void RetinexModel::computer(const cv::Mat& src, cv::Mat& reflectance, cv::Mat&  illumination)
{
	cv::Mat src_s;
	_img_cols = src.cols;
	_img_rows = src.rows;


	//// the decomposition is conducted on S channel of HSV
	//cv::cvtColor(src, src_hsv, CV_BGR2HSV);
	//std::vector<cv::Mat> V;
	//cv::split(src_hsv, V);
	src.convertTo(src_s, CV_32F);
	cv::normalize(src_s, src_s, 0, 1, CV_MINMAX);
	
	

	// initial reflectance and illumination
	illumination = src_s.clone();
	reflectance = cv::Mat::ones(src_s.size(), CV_32F);

	//optimize
	for (int i = 0; i < _param.K; i++)
	{
		//update I
		cv::Mat preI = illumination.clone();
		cv::Mat preR = reflectance.clone();
		illumination = src_s / reflectance;
		cv::Mat dx_I, dy_I;
		dx_I.create(illumination.size(), CV_32F);
		dy_I.create(illumination.size(), CV_32F);
		float* dxI_data = (float*)dx_I.data;
		float* dyI_data = (float*)dy_I.data;
		float* I_data = (float*)illumination.data;
		diff(I_data, dxI_data, dyI_data);
		cv::Mat Sx, Sy;
		weightingmatrix(dx_I, dy_I, Sx, Sy,_param.gamma_s);
		solveclosesolution(src_s, Sx, Sy, reflectance, illumination, _param.lamda_i);

		float eplison_I = FnormMatrix(illumination -preI) / FnormMatrix(preI);
		std::cout << eplison_I << std::endl;
		//update R
		reflectance = src_s / illumination;
		cv::Mat dx_R, dy_R;
		dx_R.create(reflectance.size(), CV_32F);
		dy_R.create(reflectance.size(), CV_32F);
		float* R_data = (float*)reflectance.data;
		float* dxR_data = (float*)dx_R.data;
		float* dyR_data = (float*)dy_R.data;
		
		diff(R_data, dxR_data, dyR_data);
		cv::Mat Tx, Ty;
		weightingmatrix(dx_R, dy_R, Tx, Ty,_param.gamma_t);
		solveclosesolution(src_s, Tx, Ty, illumination, reflectance,_param.lamda_r);
		float eplison_R = FnormMatrix(reflectance - preR) / FnormMatrix(preR);
		std::cout << eplison_R << std::endl;
		if (eplison_I <= _param.epsilon || eplison_R <= _param.epsilon)
		{
			break;
		}
	}
}

void RetinexModel::diff(float*& src, float*& dx, float*& dy)
{
	for (int i = 0; i < _img_rows; ++i)
	{
		for (int j = 0; j < _img_cols; ++j)
		{
			if (i < _img_rows - 1)
				dy[i*_img_cols + j] = -src[i*_img_cols + j] + src[(i + 1)*_img_cols + j];
			else
				dy[i*_img_cols + j] = 0;
			if (j < _img_cols - 1)
				dx[i*_img_cols + j] = -src[i*_img_cols + j] + src[i*_img_cols + j + 1];
			else
				dx[i*_img_cols + j] = 0;

		}
	}
		
	
	
}

void RetinexModel::weightingmatrix(cv::Mat& dx_I, cv::Mat& dy_I,cv::Mat& S0_x, cv::Mat& S0_y, float gamma)
{

	S0_x.create(dx_I.size(), CV_32F);
	S0_y.create(dy_I.size(), CV_32F);
	
	float* dxI_data = (float*)dx_I.data;
	float* dyI_data = (float*)dy_I.data;
	
	float* S0_x_data = (float*)S0_x.data;
	float* S0_y_data = (float*)S0_y.data;


	const int ru[8] = { +1, -1, +0, +0, +1, -1, -1, +1 };
	const int rv[8] = { +0, +0, +1, -1, +1, +1, -1, -1 };
	for (int i = 0; i < _img_rows; ++i)
	{
		for (int j = 0; j < _img_cols; ++j)
		{
			float temp_dxI = dxI_data[(i)*_img_cols + j ];
			float temp_dyI = dyI_data[(i)*_img_cols + j];
			float count = 1;
			
			for (int dir = 0; dir < 8; dir++)
			{
				if ((j + ru[dir]) >= 0 && (j + ru[dir]) <= _img_cols-1 && (i + rv[dir]) >= 0 && (i + rv[dir]) <= _img_rows-1)
				{
					temp_dxI = temp_dxI + dxI_data[(i + rv[dir])*_img_cols + j + ru[dir]];
					temp_dyI = temp_dyI + dyI_data[(i + rv[dir])*_img_cols + j + ru[dir]];
					count = count + 1.0;
				}
			}

			

			S0_x_data[i*_img_cols + j] = 1.0 / pow(cv::max(abs(temp_dxI / count),(float)0.01),gamma)/*pow(cv::max(abs(temp_dxI/ count),(float)0.001),-gamma)*/;
			S0_y_data[i*_img_cols + j] = 1.0 / pow(cv::max(abs(temp_dyI / count), (float)0.01), gamma) /*pow(cv::max(abs(temp_dyI /count), (float)0.001), -gamma)*/;
		}
	}
}

void RetinexModel::solveclosesolution(cv::Mat& img, cv::Mat& Pori_x, cv::Mat& pori_y, cv::Mat& fiximg, cv::Mat& output,float lamda)
{
	std::vector <Eigen::Triplet<float>> xentries;
	std::vector <Eigen::Triplet<float>> yentries;
	std::vector <Eigen::Triplet<float>> Rentries;
	Eigen::VectorXf b(_img_cols*_img_rows);
	float* src_data = (float*)img.data;
	float* S_x_data = (float*)Pori_x.data;
	float* S_y_data = (float*)pori_y.data;
	float* R_data = (float*)fiximg.data;
	for (int y = 0; y < _img_rows; ++y)
	{
		for (int x = 0; x < _img_cols; ++x)
		{
			if (x < _img_cols - 1)
			{
				xentries.push_back(Eigen::Triplet<float>(y*_img_cols + x,
					y*_img_cols + x, -1 * S_x_data[y*_img_cols + x] * lamda));
				xentries.push_back(Eigen::Triplet<float>(y*_img_cols + x,
					y*_img_cols + x + 1, +1 * S_x_data[y*_img_cols + x] * lamda));
			}
			
			if (y < _img_rows - 1)
			{
				yentries.push_back(Eigen::Triplet<float>(y*_img_cols + x,
					y*_img_cols + x, -1 * S_y_data[y*_img_cols + x] * lamda));
				yentries.push_back(Eigen::Triplet<float>(y*_img_cols + x,
					(y + 1)*_img_cols + x, +1 * S_y_data[y*_img_cols + x] * lamda));
			}
			
			Rentries.push_back(Eigen::Triplet<float>(y*_img_cols + x, y*_img_cols + x,
				R_data[y*_img_cols + x]));
			b(y*_img_cols + x) = src_data[y*_img_cols + x] * R_data[y*_img_cols + x];
		}
	}
	Eigen::SparseMatrix <float> Ux(_img_cols*_img_rows, _img_cols*_img_rows),
		Uy(_img_cols*_img_rows, _img_cols*_img_rows),
		Rvec(_img_cols*_img_rows, _img_cols*_img_rows);
	Ux.setFromTriplets(xentries.begin(), xentries.end());
	Uy.setFromTriplets(yentries.begin(), yentries.end());
	Rvec.setFromTriplets(Rentries.begin(), Rentries.end());
	Eigen::SparseMatrix <float> L = Ux.transpose()*Ux + Uy.transpose()*Uy;
	Eigen::SparseMatrix <float> RTR = Rvec.transpose()*Rvec;
	Eigen::SparseMatrix <float> A = L + RTR;

	
	Eigen::ConjugateGradient<Eigen::SparseMatrix < float >> m_solver;
	m_solver.compute(A);
	Eigen::VectorXf x = m_solver.solve(b);
	cv::Mat updateI = cv::Mat(img.size(), CV_32F, x.data());
	output = updateI.clone();
}
