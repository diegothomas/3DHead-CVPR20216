#include "stdafx.h"
#include "HeadOffV2.h"

/*******************************************************************/
/*******Functions and variables for face feature detector***********/
/*******************************************************************/
cv::Scalar BLUE(255, 0, 0);
cv::Scalar GREEN(0, 255, 0);
cv::Scalar RED(0, 0, 255);
cv::Scalar FACECOLOR(50, 255, 50);

cv::Scalar LEFTEYECOLORDOWN(255, 0, 0);
cv::Scalar LEFTEYECOLORUP(255, 255, 0);
cv::Scalar RIGHTEYECOLORDOWN(0, 255, 0);
cv::Scalar RIGHTEYECOLORUP(0, 255, 255);
cv::Scalar MOUTHCOLORDOWN(0, 0, 255);
cv::Scalar MOUTHCOLORUP(255, 0, 255);

cv::Vec3b LEFTEYEUP(255, 0, 0);
cv::Vec3b LEFTEYEDOWN(0, 255, 0);
cv::Vec3b RIGHTEYEUP(0, 0, 255);
cv::Vec3b RIGHTEYEDOWN(255, 0, 255);
cv::Vec3b LIPUP(255, 255, 0);
cv::Vec3b LIPDOWN(0, 255, 255);

bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

void drawFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, bool circle = false)
{
	int thickness = 1;
	int lineType = CV_AA;
	for (int i = start; i < end; i++) {
		line(img, cv::Point(X.at<float>(0, i), X.at<float>(1, i)),
			cv::Point(X.at<float>(0, i + 1), X.at<float>(1, i + 1)),
			FACECOLOR, thickness, lineType);
	}
	if (circle) {
		line(img, cv::Point(X.at<float>(0, end), X.at<float>(1, end)),
			cv::Point(X.at<float>(0, start), X.at<float>(1, start)),
			FACECOLOR, thickness, lineType);
	}
}

void fillFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, cv::Scalar& colordown, cv::Scalar& colorup)
{

	float scale = sqrt((X.at<float>(0, start + (end - start) / 2) - X.at<float>(0, start)) * (X.at<float>(0, start + (end - start) / 2) - X.at<float>(0, start)) +
		(X.at<float>(1, start + (end - start) / 2) - X.at<float>(1, start)) * (X.at<float>(1, start + (end - start) / 2) - X.at<float>(1, start)));

	int thickness = 1;
	int lineType = CV_AA;
	cv::Point Quad[4];
	for (int i = start; i <= end; i++) {

		//compute normal vector
		int k = i == start ? end : i - 1;
		int l = i == end ? start : i + 1;
		float nmle[2];
		nmle[0] = (X.at<float>(1, l) - X.at<float>(1, k));
		nmle[1] = -(X.at<float>(0, l) - X.at<float>(0, k));
		float val = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1]);
		nmle[0] = nmle[0] / val;
		nmle[1] = nmle[1] / val;

		Quad[0] = cv::Point(X.at<float>(0, i), X.at<float>(1, i));
		Quad[1] = cv::Point(X.at<float>(0, i) + nmle[0] * (scale / 2.0f), X.at<float>(1, i) + nmle[1] * (scale / 2.0f));

		//compute normal vector
		k = i;
		l = i + 1 > end - 1 ? start + 1 : i + 2;
		nmle[0] = (X.at<float>(1, l) - X.at<float>(1, k));
		nmle[1] = -(X.at<float>(0, l) - X.at<float>(0, k));
		val = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1]);
		nmle[0] = nmle[0] / val;
		nmle[1] = nmle[1] / val;

		if (i == end) {
			Quad[3] = cv::Point(X.at<float>(0, start), X.at<float>(1, start));
			Quad[2] = cv::Point(X.at<float>(0, start) + nmle[0] * (scale / 2.0f), X.at<float>(1, start) + nmle[1] * (scale / 2.0f));
		}
		else {
			Quad[3] = cv::Point(X.at<float>(0, i + 1), X.at<float>(1, i + 1));
			Quad[2] = cv::Point(X.at<float>(0, i + 1) + nmle[0] * (scale / 2.0f), X.at<float>(1, i + 1) + nmle[1] * (scale / 2.0f));
		}

		if (i > start + (end - start) / 2)
			fillConvexPoly(img, Quad, 4, colordown, lineType);
		else
			fillConvexPoly(img, Quad, 4, colorup, lineType);
	}
}

void drawFace(cv::Mat& img, const cv::Mat& X)
{
	//// left eyebrow
	//drawFaceHelper(img, X, 0, 4);
	//// right eyebrow
	//drawFaceHelper(img, X, 5, 9);
	//// nose
	//drawFaceHelper(img, X, 10, 13);
	//// under nose
	//drawFaceHelper(img, X, 14, 18);
	//// left eye
	//drawFaceHelper(img, X, 19, 24, true);
	//// right eye
	//drawFaceHelper(img, X, 25, 30, true);
	//// mouth contour
	//drawFaceHelper(img, X, 31, 42, true);
	//// inner mouth
	//drawFaceHelper(img, X, 43, 48, true);
	//// contour
	//if (X.cols > 49)
	//	drawFaceHelper(img, X, 49, 65);

	//// draw points
	//for (int i = 0; i < X.cols; i++)
	//	cv::circle(img, cv::Point((int)X.at<float>(0, i), (int)X.at<float>(1, i)), 1, FACECOLOR, -1);

	// Fill left eye (right for user)
	fillFaceHelper(img, X, 19, 24, LEFTEYECOLORDOWN, LEFTEYECOLORUP);
	// Fill right eye (left for user)
	fillFaceHelper(img, X, 25, 30, RIGHTEYECOLORDOWN, RIGHTEYECOLORUP);

	// Fill inner mouth
	cv::Mat Mouth = cv::Mat::zeros(2, 8, CV_32F);
	Mouth.at<float>(0, 0) = X.at<float>(0, 31); Mouth.at<float>(1, 0) = X.at<float>(1, 31);
	Mouth.at<float>(0, 1) = X.at<float>(0, 43); Mouth.at<float>(1, 1) = X.at<float>(1, 43);
	Mouth.at<float>(0, 2) = X.at<float>(0, 44); Mouth.at<float>(1, 2) = X.at<float>(1, 44);
	Mouth.at<float>(0, 3) = X.at<float>(0, 45); Mouth.at<float>(1, 3) = X.at<float>(1, 45);
	Mouth.at<float>(0, 4) = X.at<float>(0, 37); Mouth.at<float>(1, 4) = X.at<float>(1, 37);
	Mouth.at<float>(0, 5) = X.at<float>(0, 46); Mouth.at<float>(1, 5) = X.at<float>(1, 46);
	Mouth.at<float>(0, 6) = X.at<float>(0, 47); Mouth.at<float>(1, 6) = X.at<float>(1, 47);
	Mouth.at<float>(0, 7) = X.at<float>(0, 48); Mouth.at<float>(1, 7) = X.at<float>(1, 48);
	fillFaceHelper(img, Mouth, 0, 7, MOUTHCOLORDOWN, MOUTHCOLORUP);
}

void drawFacePoints(cv::Mat& img, const cv::Mat& X)
{
	//// left eyebrow
	//drawFaceHelper(img, X, 0, 4);
	//// right eyebrow
	//drawFaceHelper(img, X, 5, 9);
	//// nose
	//drawFaceHelper(img, X, 10, 13);
	//// under nose
	//drawFaceHelper(img, X, 14, 18);
	//// left eye
	//drawFaceHelper(img, X, 19, 24, true);
	//// right eye
	//drawFaceHelper(img, X, 25, 30, true);
	//// mouth contour
	//drawFaceHelper(img, X, 31, 42, true);
	//// inner mouth
	//drawFaceHelper(img, X, 43, 48, true);
	//// contour
	//if (X.cols > 49)
	//	drawFaceHelper(img, X, 49, 65);

	// draw points
	for (int i = 0; i < X.cols; i++)
		cv::circle(img, cv::Point((int)X.at<float>(0, i), (int)X.at<float>(1, i)), 1, FACECOLOR, -1);
}

void drawPose(cv::Mat& img, const cv::Mat& rot)
{
	int loc[2] = { 70, 70 };
	int thickness = 2;
	int lineType = CV_AA;
	float lineL = 50.f;

	cv::Mat P = (cv::Mat_<float>(3, 4) <<
		0, lineL, 0, 0,
		0, 0, -lineL, 0,
		0, 0, 0, -lineL);
	P = rot.rowRange(0, 2)*P;
	P.row(0) += loc[0];
	P.row(1) += loc[1];
	cv::Point p0(P.at<float>(0, 0), P.at<float>(1, 0));

	line(img, p0, cv::Point(P.at<float>(0, 1), P.at<float>(1, 1)), BLUE, thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 2), P.at<float>(1, 2)), GREEN, thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 3), P.at<float>(1, 3)), RED, thickness, lineType);
}


// Implement Paraleele Relaxation algorithm 2.1. in paper [A parallel relaxation method for quadratic programming problems with interval constraints]
Eigen::VectorXd ParallelRelaxation(Eigen::MatrixXd Q_inv, Eigen::VectorXd x0, Eigen::VectorXd lb, Eigen::VectorXd ub) {
	int n = x0.size();
	double w = 0.02;
	int max_iter = 1000;// 30000;
	Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd Delta = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd Gamma = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd xres = x0;
	Eigen::VectorXd S = Eigen::VectorXd::Zero(n);

	/*double theta;
	double w_min = 1.0;
	for (int i = 0; i < n; i++) {
		double val = 0.0;
		for (int j = 0; j < n; j++) {
			if (i == j)
				continue;
			val += abs(Q_inv(i, j));
		}
		theta = (2.0 / Q_inv(i, i)) * val;
		double w_curr = min(1 / theta, 3.0 / (2.0 + theta));
		w_min = min(w_min, w_curr);
	}
	cout << "w_max: " <<  w_min << endl;*/

	bool converged = false;
	int iter = 0;
	while (!converged) {

		for (int i = 0; i < n; i++) {
			Delta(i) = (lb(i) - xres(i)) / Q_inv(i, i);
			Gamma(i) = (ub(i) - xres(i)) / Q_inv(i, i);
			S(i) = MyMedian(u(i), w*Delta(i), w*Gamma(i));
		}
		u = u - S;

		xres = xres + Q_inv * S;

		iter++;
		converged = (iter > max_iter || S.norm() < 1.0e-10);
	}

	//cout << "number of inner loops: " << iter << "S.norm(): " << S.norm() << endl;
	return xres;
}

/*******************************************************************/
/****END**Functions and variables for face feature detector*********/
/*******************************************************************/

cl_kernel LoadKernel(string filename, string Kernelname, cl_context context, cl_device_id device) {
	cl_int ret;
	std::ifstream file(filename);
	checkErr(file.is_open() ? CL_SUCCESS : -1, filename.c_str());
	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	const char * code = prog.c_str();
	cl_program lProgram = clCreateProgramWithSource(context, 1, &code, 0, &ret);
	ret = clBuildProgram(lProgram, 1, &device, "", 0, 0);
	checkErr(ret, "Program::build()");

	cl_kernel kernel = clCreateKernel(lProgram, Kernelname.c_str(), &ret);
	checkErr(ret, (Kernelname + string("::Kernel()")).c_str());
	return kernel;
}

void GetWeightedNormal(MyMesh *TheMesh, Face *triangle, float *nmle) {

	double v1[3];
	double v2[3];
	double b[3];
	double h[3];

	v1[0] = double(TheMesh->_vertices[triangle->_v2]->_x - TheMesh->_vertices[triangle->_v1]->_x);
	v1[1] = double(TheMesh->_vertices[triangle->_v2]->_y - TheMesh->_vertices[triangle->_v1]->_y);
	v1[2] = double(TheMesh->_vertices[triangle->_v2]->_z - TheMesh->_vertices[triangle->_v1]->_z);

	v2[0] = double(TheMesh->_vertices[triangle->_v3]->_x - TheMesh->_vertices[triangle->_v1]->_x);
	v2[1] = double(TheMesh->_vertices[triangle->_v3]->_y - TheMesh->_vertices[triangle->_v1]->_y);
	v2[2] = double(TheMesh->_vertices[triangle->_v3]->_z - TheMesh->_vertices[triangle->_v1]->_z);

	double nrm = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	b[0] = v1[0] / nrm;
	b[1] = v1[1] / nrm;
	b[2] = v1[2] / nrm;
	double proj = b[0] * v2[0] + b[1] * v2[1] + b[2] * v2[2];
	h[0] = v2[0] - proj*b[0];
	h[1] = v2[1] - proj*b[1];
	h[2] = v2[2] - proj*b[2];
	double hauteur = sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
	double area = nrm * hauteur / 2.0;

	double tmp[3];
	tmp[0] = v1[1] * v2[2] - v1[2] * v2[1];
	tmp[1] = -v1[0] * v2[2] + v1[2] * v2[0];
	tmp[2] = v1[0] * v2[1] - v1[1] * v2[0];

	nrm = sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2]);
	nmle[0] = float(tmp[0] / nrm)*area;
	nmle[1] = float(tmp[1] / nrm)*area;
	nmle[2] = float(tmp[2] / nrm)*area;

}

void getWeightsB(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3) {
	if (u == uv1._u && v == uv1._v) {
		weights[0] = 1.0f;
		weights[1] = 0.0f;
		weights[2] = 0.0f;
		return;
	}
	if (u == uv2._u && v == uv2._v) {
		weights[0] = 0.0f;
		weights[1] = 1.0f;
		weights[2] = 0.0f;
		return;
	}
	if (u == uv3._u && v == uv3._v) {
		weights[0] = 0.0f;
		weights[1] = 0.0f;
		weights[2] = 1.0f;
		return;
	}

	// test if flat triangle
	TextUV e0 = uv2 - uv1;
	TextUV e1 = uv3 - uv1;

	if ((e0._u*e1._v - e0._v*e1._u) == 0.0f) { // flat triangle
		// find two extrema
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v2 and v3 since e0 and e1 are in opposite direction
			weights[0] = 0.0f;
			weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
			weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
			return;
		}

		e0 = uv1 - uv2;
		e1 = uv3 - uv2;
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v1 and v3 since e0 and e1 are in opposite direction
			weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
			weights[1] = 0.0f;
			weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
			return;
		}

		e0 = uv1 - uv3;
		e1 = uv2 - uv3;
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v1 and v2 since e0 and e1 are in opposite direction
			weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
			weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
			weights[2] = 0.0f;
			return;
		}

		cout << "Missing case!!" << endl;
		return;
	}

	double A[2];
	double tmp;
	double B[2];

	// Compute percentage of ctrl point 1
	A[0] = uv3._u - uv2._u;
	A[1] = uv3._v - uv2._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv1._u;
	B[1] = v - uv1._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	double num, den, lambda;
	if (B[0] != 0.0) {
		num = uv1._v - uv2._v + (uv2._u - uv1._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv1._u - uv2._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 0" << endl;
		return;
		//}
	}


	double inter_pos[2];
	inter_pos[0] = uv2._u + lambda*A[0];
	inter_pos[1] = uv2._v + lambda*A[1];

	A[0] = inter_pos[0] - uv1._u;
	A[1] = inter_pos[1] - uv1._v;
	double val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[0] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[0] = val == 0.0 ? 0.0f : float(weights[0] / val);

	// Compute percentage of ctrl point 2
	A[0] = uv1._u - uv3._u;
	A[1] = uv1._v - uv3._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv2._u;
	B[1] = v - uv2._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	if (B[0] != 0.0) {
		num = uv2._v - uv3._v + (uv3._u - uv2._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv2._u - uv3._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 1" << endl;
		//	cout << "u v " << u << " " << v << endl;
		//	cout << "u1 v1 " << uv1._u << " " << uv1._v << endl;
		//	cout << "u2 v2 " << uv2._u << " " << uv2._v << endl;
		//	cout << "u3 v3 " << uv3._u << " " << uv3._v << endl;
		//	cout << "A " << A[0] << " " << A[1] << endl;
		//	cout << "B " << B[0] << " " << B[1] << endl;
		//	weights[0] = -1.0f;
		//	int tmp;
		//	cin >> tmp;
		//	weights[0] = -1.0f;
		return;
		//}
	}

	inter_pos[0] = uv3._u + lambda*A[0];
	inter_pos[1] = uv3._v + lambda*A[1];

	A[0] = inter_pos[0] - uv2._u;
	A[1] = inter_pos[1] - uv2._v;
	val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[1] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[1] = val == 0.0 ? 0.0f : float(weights[1] / val);

	// Compute percentage of ctrl point 3
	A[0] = uv1._u - uv2._u;
	A[1] = uv1._v - uv2._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv3._u;
	B[1] = v - uv3._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	if (B[0] != 0.0) {
		num = uv3._v - uv2._v + (uv2._u - uv3._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv3._u - uv2._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 2" << endl;
		//weights[0] = -1.0f;
		return;
		//}
	}

	inter_pos[0] = uv2._u + lambda*A[0];
	inter_pos[1] = uv2._v + lambda*A[1];

	A[0] = inter_pos[0] - uv3._u;
	A[1] = inter_pos[1] - uv3._v;
	val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[2] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[2] = val == 0.0 ? 0.0f : float(weights[2] / val);
	return;
}

void getWeights(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3)  {
	weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
	weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
	weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
	return;
}

void DrawTriangle(MyMesh *TheMesh, Face *triangle, float *data, USHORT color) {
	TextUV v0 = TextUV(round(TheMesh->_uvs[triangle->_t1]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t1]->_v*float(BumpWidth)));
	TextUV v1 = TextUV(round(TheMesh->_uvs[triangle->_t2]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t2]->_v*float(BumpWidth)));
	TextUV v2 = TextUV(round(TheMesh->_uvs[triangle->_t3]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t3]->_v*float(BumpWidth)));

	//Compute the three edges of the triangle and line equation
	TextUV e0 = v1 - v0;
	float lambda0 = (v1._u - v0._u) == 0.0 ? 1.0e12 : (v1._v - v0._v) / (v1._u - v0._u);
	float c0 = lambda0 > 1.0e10 ? 0.0 : v0._v - lambda0*v0._u;
	TextUV e1 = v2 - v0;
	float lambda1 = (v2._u - v0._u) == 0.0 ? 1.0e12 : (v2._v - v0._v) / (v2._u - v0._u);
	float c1 = lambda1 > 1.0e10 ? 0.0 : v0._v - lambda1*v0._u;
	TextUV e2 = v2 - v1;
	float lambda2 = (v2._u - v1._u) == 0.0 ? 1.0e12 : (v2._v - v1._v) / (v2._u - v1._u);
	float c2 = lambda2 > 1.0e10 ? 0.0 : v1._v - lambda2*v1._u;

	int min_u = int(round(min(v0._u, min(v1._u, v2._u)))); // get min of vertical coordinates (= y coordinates)
	int max_u = int(round(max(v0._u, max(v1._u, v2._u)))); // get max of vertical coordinates (= y coordinates)

	// draw triangle line by line
	for (int i = min_u; i < max_u + 1; i++) {
		// Compute intersection between line {y=i (i.e. u= i)} and all 3 edges
		TextUV inter0 = TextUV(i, lambda0*i + c0);
		if (lambda0 > 1.0e10) {
			inter0._u = -1.0;
			inter0._v = -1.0;
		}
		TextUV inter1 = TextUV(i, lambda1*i + c1);
		if (lambda1 > 1.0e10) {
			inter1._u = -1.0;
			inter1._v = -1.0;
		}
		TextUV inter2 = TextUV(i, lambda2*i + c2);
		if (lambda2 > 1.0e10) {
			inter2._u = -1.0;
			inter2._v = -1.0;
		}

		// identify valid intersections
		bool valid0 = true;
		bool valid1 = true;
		bool valid2 = true;
		if (((inter0 - v0)._u*e0._u + (inter0 - v0)._v*e0._v) < 0.0 || ((inter0 - v1)._u*e0._u + (inter0 - v1)._v*e0._v) > 0.0 || inter0._u == -1) {
			valid0 = false;
		}
		if (((inter1 - v0)._u*e1._u + (inter1 - v0)._v*e1._v) < 0.0 || ((inter1 - v2)._u*e1._u + (inter1 - v2)._v*e1._v) > 0.0 || inter1._u == -1) {
			valid1 = false;
		}
		if (((inter2 - v1)._u*e2._u + (inter2 - v1)._v*e2._v) < 0.0 || ((inter2 - v2)._u*e2._u + (inter2 - v2)._v*e2._v) > 0.0 || inter2._u == -1) {
			valid2 = false;
		}

		int min_v = BumpWidth;
		int max_v = 0;

		if (valid0) {
			min_v = min(min_v, int(round(inter0._v)));
			max_v = max(max_v, int(round(inter0._v)));
		}

		if (valid1) {
			min_v = min(min_v, int(round(inter1._v)));
			max_v = max(max_v, int(round(inter1._v)));
		}

		if (valid2) {
			min_v = min(min_v, int(round(inter2._v)));
			max_v = max(max_v, int(round(inter2._v)));
		}

		for (int j = min_v; j < max_v + 1; j++) {
			data[4 * (i*BumpWidth + j) + 2] = color;
		}

	}

	return;
}

bool IsInTriangle(MyMesh *TheMesh, Face *triangle, int i, int j) {
	TextUV v0 = TextUV(TheMesh->_uvs[triangle->_t1]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t1]->_v*float(BumpWidth));
	TextUV v1 = TextUV(TheMesh->_uvs[triangle->_t2]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t2]->_v*float(BumpWidth));
	TextUV v2 = TextUV(TheMesh->_uvs[triangle->_t3]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t3]->_v*float(BumpWidth));
	TextUV v = TextUV(float(i), float(j));

	TextUV e0 = v0 - v;
	TextUV e1 = v1 - v;
	TextUV e2 = v2 - v;


	float p0 = e0._u*e1._v - e0._v*e1._u;
	float p1 = e1._u*e2._v - e1._v*e2._u;
	float p2 = e2._u*e0._v - e2._v*e0._u;

	return (p0 >= 0.0f && p1 >= 0.0f && p2 >= 0.0f) || (p0 <= 0.0f && p1 <= 0.0f && p2 <= 0.0f);
}

void HeadOffV2::StartKinect(){
	/* Check for Kinect */
	HRESULT hr;
	_Ske = new SkeletonTrack();

	if ((hr = _Ske->initKinect() != S_OK)) {
		cout << "Error initKinect" << endl;
	}

	assert(_Ske->getCheight() == height);
	assert(_Ske->getCwidth() == width);
}

void HeadOffV2::StartKinect2(){
	/* Check for Kinect V2 */
	_KinectV2 = new KinectV2Manager();
	_KinectV2->InitializeDefaultSensor();
}

void HeadOffV2::Draw(bool color, bool bump) {

	//glBegin(GL_POINTS);
	glColor4f(1.0, 1.0, 1.0, 1.0);

	float pt[3];
	float nmle[3];
	Point3DGPU *currV;
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			if (_Bump.at<cv::Vec4f>(i, j)[1] > 0.0) {
				currV = &_verticesBump[i*BumpWidth + j];

				// Transform points to match tracking
				if (bump) {
					pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
					pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
					pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);
					
					//pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
					//pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
					//pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];

					nmle[0] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(0, 2);
					nmle[1] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(1, 2);
					nmle[2] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(2, 2);
				}
				else {
					pt[0] = currV->_x * _Rotation_inv(0, 0) + currV->_y * _Rotation_inv(0, 1) + currV->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
					pt[1] = currV->_x * _Rotation_inv(1, 0) + currV->_y * _Rotation_inv(1, 1) + currV->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
					pt[2] = currV->_x * _Rotation_inv(2, 0) + currV->_y * _Rotation_inv(2, 1) + currV->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

					nmle[0] = currV->_Nx * _Rotation_inv(0, 0) + currV->_Ny * _Rotation_inv(0, 1) + currV->_Nz * _Rotation_inv(0, 2);
					nmle[1] = currV->_Nx * _Rotation_inv(1, 0) + currV->_Ny * _Rotation_inv(1, 1) + currV->_Nz * _Rotation_inv(1, 2);
					nmle[2] = currV->_Nx * _Rotation_inv(2, 0) + currV->_Ny * _Rotation_inv(2, 1) + currV->_Nz * _Rotation_inv(2, 2);
				}

				if (_RGBMapBump.at<cv::Vec4f>(i, j)[0] == 255.0f && _RGBMapBump.at<cv::Vec4f>(i, j)[1] == 0.0f && _RGBMapBump.at<cv::Vec4f>(i, j)[2] == 0.0f)
					glPointSize(3.0);

				glBegin(GL_POINTS);
				if (color) {
					glColor4f(_RGBMapBump.at<cv::Vec4f>(i, j)[0] / 255.0, _RGBMapBump.at<cv::Vec4f>(i, j)[1] / 255.0, _RGBMapBump.at<cv::Vec4f>(i, j)[2] / 255.0, 1.0);
				}
				glNormal3f(nmle[0], nmle[1], nmle[2]);
				glVertex3f(pt[0], pt[1], pt[2]);
				glEnd();
				glPointSize(1.0);
				/*glNormal3f(currV->_TNx, currV->_TNy, currV->_TNz);
				glVertex3f(currV->_Tx, currV->_Ty, currV->_Tz);*/
			}
		}
	}

	//glEnd();
}

void HeadOffV2::DrawBlendedMesh(vector<MyMesh *> Blendshape) {
	MyMesh *RefMesh = Blendshape[0];

	glColor4f(1.0, 1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);

	float tmpPt[3];
	float tmpNmle[3];
	MyPoint *s1, *s2, *s3;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		s1 = RefMesh->_vertices[(*it)->_v1];
		s2 = RefMesh->_vertices[(*it)->_v2];
		s3 = RefMesh->_vertices[(*it)->_v3];

		tmpPt[0] = s1->_x;
		tmpPt[1] = s1->_y;
		tmpPt[2] = s1->_z;
		tmpNmle[0] = s1->_Nx;
		tmpNmle[1] = s1->_Ny;
		tmpNmle[2] = s1->_Nz;

		for (int i = 1; i < 28; i++) {
			tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s1->_x - Blendshape[i]->_vertices[(*it)->_v1]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s1->_y - Blendshape[i]->_vertices[(*it)->_v1]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s1->_z - Blendshape[i]->_vertices[(*it)->_v1]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s1->_Nx - Blendshape[i]->_vertices[(*it)->_v1]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s1->_Ny - Blendshape[i]->_vertices[(*it)->_v1]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s1->_Nz - Blendshape[i]->_vertices[(*it)->_v1]->_Nz);
		}
		float nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);

		tmpPt[0] = s2->_x;
		tmpPt[1] = s2->_y;
		tmpPt[2] = s2->_z;
		tmpNmle[0] = s2->_Nx;
		tmpNmle[1] = s2->_Ny;
		tmpNmle[2] = s2->_Nz;

		for (int i = 1; i < 28; i++) {
			tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s2->_x - Blendshape[i]->_vertices[(*it)->_v2]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s2->_y - Blendshape[i]->_vertices[(*it)->_v2]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s2->_z - Blendshape[i]->_vertices[(*it)->_v2]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s2->_Nx - Blendshape[i]->_vertices[(*it)->_v2]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s2->_Ny - Blendshape[i]->_vertices[(*it)->_v2]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s2->_Nz - Blendshape[i]->_vertices[(*it)->_v2]->_Nz);
		}
		nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);


		tmpPt[0] = s3->_x;
		tmpPt[1] = s3->_y;
		tmpPt[2] = s3->_z;
		tmpNmle[0] = s3->_Nx;
		tmpNmle[1] = s3->_Ny;
		tmpNmle[2] = s3->_Nz;

		for (int i = 1; i < 28; i++) {
			tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s3->_x - Blendshape[i]->_vertices[(*it)->_v3]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s3->_y - Blendshape[i]->_vertices[(*it)->_v3]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s3->_z - Blendshape[i]->_vertices[(*it)->_v3]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s3->_Nx - Blendshape[i]->_vertices[(*it)->_v3]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s3->_Ny - Blendshape[i]->_vertices[(*it)->_v3]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s3->_Nz - Blendshape[i]->_vertices[(*it)->_v3]->_Nz);
		}
		nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);
	}

	glEnd();
}

void HeadOffV2::DrawQuad(int color, bool bump) {

	glBegin(GL_QUADS);
	glColor4f(1.0, 1.0, 1.0, 1.0);

	int indx_i[4] = { 0, 1, 1, 0 };
	int indx_j[4] = { 0, 0, 1, 1 };

	float pt[3];
	float nmle[3];
	Point3DGPU *currV;
	for (int i = 0; i < BumpHeight-1; i++) {
		for (int j = 0; j < BumpWidth-1; j++) {
			//if (_Bump.at<cv::Vec4f>(i, j)[1] > 0.0f && _Bump.at<cv::Vec4f>(i + 1, j)[1] > 0.0f && _Bump.at<cv::Vec4f>(i + 1, j + 1)[1] > 0.0f && _Bump.at<cv::Vec4f>(i, j + 1)[1] > 0.0f) {
			if (_VMapBump.at<cv::Vec4f>(i, j)[0] != 0.0f && _VMapBump.at<cv::Vec4f>(i + 1, j)[0] != 0.0f &&_VMapBump.at<cv::Vec4f>(i + 1, j + 1)[0] != 0.0f && _VMapBump.at<cv::Vec4f>(i, j + 1)[0] != 0.0f) {
				for (int k = 0; k < 4; k++) {
					currV = &_verticesBump[(i + indx_i[k])*BumpWidth + (j + indx_j[k])];

					// Transform points to match tracking
					if (bump) {
						pt[0] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
						pt[1] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
						pt[2] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

						//pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
						//pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
						//pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];

						nmle[0] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(0, 2);
						nmle[1] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(1, 2);
						nmle[2] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(2, 2);
					}
					else {
						pt[0] = currV->_x * _Rotation_inv(0, 0) + currV->_y * _Rotation_inv(0, 1) + currV->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
						pt[1] = currV->_x * _Rotation_inv(1, 0) + currV->_y * _Rotation_inv(1, 1) + currV->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
						pt[2] = currV->_x * _Rotation_inv(2, 0) + currV->_y * _Rotation_inv(2, 1) + currV->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

						nmle[0] = currV->_Nx * _Rotation_inv(0, 0) + currV->_Ny * _Rotation_inv(0, 1) + currV->_Nz * _Rotation_inv(0, 2);
						nmle[1] = currV->_Nx * _Rotation_inv(1, 0) + currV->_Ny * _Rotation_inv(1, 1) + currV->_Nz * _Rotation_inv(1, 2);
						nmle[2] = currV->_Nx * _Rotation_inv(2, 0) + currV->_Ny * _Rotation_inv(2, 1) + currV->_Nz * _Rotation_inv(2, 2);
					}

					if (color == 0) {
						glColor4f((nmle[0] + 1.0) / 2.0, (nmle[1] + 1.0) / 2.0, (nmle[2] + 1.0) / 2.0, 1.0);
					}
					if (color == 1) {
						glColor4f(_RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] / 255.0, 1.0);
					}
					glNormal3f(nmle[0], nmle[1], nmle[2]);
					glVertex3f(pt[0], pt[1], pt[2]);
					//if (pt[2] > -0.5f) {
					//	cout << "pt: " << pt[0] << ", " << pt[1] << ", " << pt[2] << "; bump: " << _Bump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] << ", " << _Bump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] << endl;
					//	cout << "VMap: " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] << ", " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] << ", " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] << endl;
					//}
				}
			}
		}
	}

	glEnd();
}

void HeadOffV2::DrawRect(bool color) {


	//glBegin(GL_POINTS);
	glColor4f(1.0, 1.0, 1.0, 1.0);

	//MyPoint *currPt = _vertices[(_pRect.y + _pRect.height / 2)*cDepthWidth + (_pRect.x + _pRect.width / 2)];
	//unsigned short ref_depth = currPt == NULL ? 65000 : currPt->_z;

	int i, j;
	for (i = _pRect.y; i < _pRect.y + _pRect.height; i++) {
		for (j = _pRect.x; j < _pRect.x + _pRect.width; j++) {
			/*currPt = _vertices[i*cDepthWidth + j];
			if (currPt == NULL)
			continue;*/

			//if (-currPt->_z > ref_depth + 0.5)
			//	continue;

			if (_RGBMap.at<cv::Vec4b>(i, j)[2] == 0 && _RGBMap.at<cv::Vec4b>(i, j)[1] == 255 && _RGBMap.at<cv::Vec4b>(i, j)[0] == 0)
				glPointSize(3.0);
			glBegin(GL_POINTS);
			if (color)
				glColor3f(float(_RGBMap.at<cv::Vec4b>(i, j)[2]) / 255.0f, float(_RGBMap.at<cv::Vec4b>(i, j)[1]) / 255.0f, float(_RGBMap.at<cv::Vec4b>(i, j)[0]) / 255.0f);
			else
				glColor3f((_NMap.at<cv::Vec4f>(i, j)[0] + 1.0f) / 2.0f, (_NMap.at<cv::Vec4f>(i, j)[1] + 1.0f) / 2.0f, (_NMap.at<cv::Vec4f>(i, j)[2] + 1.0f) / 2.0f);
			glNormal3f(_NMap.at<cv::Vec4f>(i, j)[0], _NMap.at<cv::Vec4f>(i, j)[1], _NMap.at<cv::Vec4f>(i, j)[2]);
			glVertex3f(_VMap.at<cv::Vec4f>(i, j)[0], _VMap.at<cv::Vec4f>(i, j)[1], _VMap.at<cv::Vec4f>(i, j)[2]);
			glEnd();
			glPointSize(1.0);
			/*glNormal3f(currPt->_Nx, currPt->_Ny, currPt->_Nz);
			glVertex3f(currPt->_x, currPt->_y, currPt->_z);
			glColor3f(currPt->_R, currPt->_G, currPt->_B);*/
		}
	}

	//glEnd();
}

void HeadOffV2::DrawLandmark(int i){
	if (_landmarks.at<float>(0, i) == 0.0f && _landmarks.at<float>(1, i) == 0.0f) {
		cout << "landmark null" << endl;
		return;
	}
	/*cout << "rows: " << _landmarks.rows << endl;
	cout << "cols: " << _landmarks.cols << endl;*/

	MyPoint *LandMark = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2]);
	if (LandMark->_x == 0.0f && LandMark->_y == 0.0f && LandMark->_z == 0.0f) {
		glEnd();
		delete LandMark;
		cout << "landmark VMAP null" << endl;
		return;
	}

	glBegin(GL_POINTS);
	glColor4f(1.0, 0.0, 0.0, 1.0);
	glVertex3f(LandMark->_x, LandMark->_y, LandMark->_z);

	int u = _landmarksBump.at<int>(0, i);
	int v = _landmarksBump.at<int>(1, i);

	if (u < 0 || u > BumpHeight - 1 || v < 0 || v > BumpWidth - 1) {
		glEnd();
		delete LandMark;
		cout << "Out of bound" << endl;
		return;
	}

	if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f) {
		glEnd();
		delete LandMark;
		cout << "Mask null" << endl;
		return;
	}

	float pt[3];
	pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
	pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
	pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

	/*pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0];
	pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[1];
	pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[2];*/

	/*float ptRef[3];
	ptRef[0] = _Vtx[0][3 * i]; ptRef[1] = _Vtx[0][3 * i + 1]; ptRef[2] = _Vtx[0][3 * i + 2];

	for (int k = 1; k < 28; k++) {
		pt[0] = pt[0] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i] - ptRef[0]);
		pt[1] = pt[1] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 1] - ptRef[1]);
		pt[2] = pt[2] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 2] - ptRef[2]);
	}
	pt[0] = pt[0] + _Translation_inv(0);
	pt[1] = pt[1] + _Translation_inv(1);
	pt[2] = pt[2] + _Translation_inv(2);*/

	glColor4f(0.0, 1.0, 0.0, 1.0);
	glVertex3f(pt[0], pt[1], pt[2]);

	glEnd();
	delete LandMark;

}

int HeadOffV2::Load() {
	//if (_idx > 950)
	//	return 0;

	char filename_buff[100];
	cv::Mat depth;
	cv::Mat color;
	cv::Mat color_origin;
	cv::Mat color_coord;

	if (_Kinect1) {
		_Ske->getKinectData();
		depth = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC3);
		color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

		BYTE* h_colorFrame = _Ske->getColorframe();
		USHORT* h_depthFrame = _Ske->getDepthframe();
		LONG* h_colorCoord = _Ske->getColorCoord();

		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				depth.at<cv::Vec3w>(i, j)[0] = h_depthFrame[i*cDepthWidth + j];
				depth.at<cv::Vec3w>(i, j)[1] = h_depthFrame[i*cDepthWidth + j];
				depth.at<cv::Vec3w>(i, j)[2] = h_depthFrame[i*cDepthWidth + j];

				LONG colorForDepthX = h_colorCoord[(i*cDepthWidth + j) * 2];
				LONG colorForDepthY = h_colorCoord[(i*cDepthWidth + j) * 2 + 1];

				// check if the color coordinates lie within the range of the color map
				if (colorForDepthX >= 0 && colorForDepthX < cDepthWidth && colorForDepthY >= 0 && colorForDepthY < cDepthHeight)
				{
					color.at<cv::Vec3b>(i, j)[0] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4];
					color.at<cv::Vec3b>(i, j)[1] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4 + 1];
					color.at<cv::Vec3b>(i, j)[2] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4 + 2];
				}
			}
		}
	}

	if (_Kinect2) {
		//cout << "read kinect v2" << endl;
		if (!_KinectV2->Available())
			return 1;

		_KinectV2->UpdateManager();

		depth = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC3);
		color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);
		color.setTo(0);

		RGBQUAD* colorFrame = _KinectV2->getColorframe();
		UINT16* depthFrame = _KinectV2->getDepthframe();
		ColorSpacePoint* colorCoord = _KinectV2->getColorCoord();
		CameraSpacePoint *cameraCoord = _KinectV2->getCameraCoord();

		CameraSpacePoint pos;
		ColorSpacePoint color_indx;
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				if (depthFrame[cDepthWidth*i + j] == 0)
					continue;

				pos = cameraCoord[cDepthWidth*i + j];

				//depth.at<cv::Vec3w>(i, j)[0] = unsigned short((pos.X + 5.0f)*6000.0f); // unsigned short(pos.Z*10000.0f);
				//depth.at<cv::Vec3w>(i, j)[1] = unsigned short((pos.Y + 5.0f)*6000.0f); // unsigned short(pos.Z*10000.0f);
				//depth.at<cv::Vec3w>(i, j)[2] = unsigned short(-pos.Z*10000.0f); // unsigned short(pos.Z*10000.0f);

				//color_indx = colorCoord[cDepthWidth*i + j];


				//int colorForDepthX = Myround(color_indx.Y);
				//int colorForDepthY = Myround(color_indx.X);

				//// check if the color coordinates lie within the range of the color map
				//if (colorForDepthX >= 0 && colorForDepthX < cColorHeight - 1 && colorForDepthY >= 0 && colorForDepthY < cColorWidth - 1)
				//{
				//	color.at<cv::Vec3b>(i, j)[2] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbRed;
				//	color.at<cv::Vec3b>(i, j)[1] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbGreen;
				//	color.at<cv::Vec3b>(i, j)[0] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbBlue;
				//}

				//Project point into virtual camera
				int p_indx[2];
				p_indx[0] = cDepthHeight - 1 - int((pos.Y / fabs(pos.Z))*_intrinsic[1] + _intrinsic[3]);
				p_indx[1] = int((pos.X / fabs(pos.Z))*_intrinsic[0] + _intrinsic[2]);

				if (p_indx[0] >= 0 && p_indx[0] < cDepthHeight && p_indx[1] >= 0 && p_indx[1] < cDepthWidth /*&& pos.Z < 1.5*/)
				{
					if (depth.at<cv::Vec3w>(p_indx[0], p_indx[1])[0] != 0 && depth.at<cv::Vec3w>(p_indx[0], p_indx[1])[0] < unsigned short(pos.Z*10000.0f))
						continue;

					depth.at<cv::Vec3w>(p_indx[0], p_indx[1])[0] = unsigned short(pos.Z*10000.0f);
					depth.at<cv::Vec3w>(p_indx[0], p_indx[1])[1] = unsigned short(pos.Z*10000.0f);
					depth.at<cv::Vec3w>(p_indx[0], p_indx[1])[2] = unsigned short(pos.Z*10000.0f);

					color_indx = colorCoord[cDepthWidth*i + j];

					int colorForDepthX = Myround(color_indx.Y);
					int colorForDepthY = Myround(color_indx.X);

					// check if the color coordinates lie within the range of the color map
					if (colorForDepthX >= 0 && colorForDepthX < cColorHeight - 1 && colorForDepthY >= 0 && colorForDepthY < cColorWidth - 1)
					{
						color.at<cv::Vec3b>(p_indx[0], p_indx[1])[2] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbRed;
						color.at<cv::Vec3b>(p_indx[0], p_indx[1])[1] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbGreen;
						color.at<cv::Vec3b>(p_indx[0], p_indx[1])[0] = colorFrame[colorForDepthX*cColorWidth + colorForDepthY].rgbBlue;
					}
				}
			}
		}
	}

	if (!_Kinect1 && !_Kinect2) {
		sprintf_s(filename_buff, "%s\\Depth_%d.tiff", _path, _idx);
		depth = cv::imread(string(filename_buff), CV_LOAD_IMAGE_UNCHANGED);
		if (!depth.data)
			return 3;

		sprintf_s(filename_buff, "%s\\RGB_%d.tiff", _path, _idx);
		color = cv::imread(string(filename_buff), CV_LOAD_IMAGE_UNCHANGED);
		if (!color.data)
			return 3;

	}


	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			_imgD.at<cv::Vec4w>(i, j)[0] = depth.at<cv::Vec3w>(i, j)[0];
			_imgD.at<cv::Vec4w>(i, j)[1] = depth.at<cv::Vec3w>(i, j)[1];
			_imgD.at<cv::Vec4w>(i, j)[2] = depth.at<cv::Vec3w>(i, j)[2];
			_imgD.at<cv::Vec4w>(i, j)[3] = 0;
			_imgS.at<cv::Vec4b>(i, j)[0] = color.at<cv::Vec3b>(i, j)[0];
			_imgS.at<cv::Vec4b>(i, j)[1] = color.at<cv::Vec3b>(i, j)[1];
			_imgS.at<cv::Vec4b>(i, j)[2] = color.at<cv::Vec3b>(i, j)[2];
			_imgS.at<cv::Vec4b>(i, j)[3] = 0;
		}
	}
	_imgS.copyTo(_imgC);

	/*cv::imshow("Color image", color);
	cv::imshow("Depth image", depth);
	cv::waitKey(1);*/

	/*char filename[100];
	sprintf_s(filename, "Seq\\KinectV1-2\\RGB_%d.tiff", _idx);
	cv::imwrite(filename, color);
	sprintf_s(filename, "Seq\\KinectV1-2\\Depth_%d.tiff", _idx);
	cv::imwrite(filename, depth);*/

	_idx++;
	return 2;
}

int HeadOffV2::LoadToSave(int k) {
	if (_ptsQ.empty())
		return -1;

	/*if (_idx_curr < 10) {
		_idx_curr++;
		return 0;
	}*/

	_depth.front().copyTo(_depth_in[k]);
	_color.front().copyTo(_color_in[k]);

	_idx_thread[k] = _idx_curr;

	_idx_curr++;
	return 1;
}

int HeadOffV2::SaveData(int k) {

	cv::Mat depth;
	cv::Mat color;
	depth = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC1);
	color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			//depth.at<cv::Vec3w>(i, j)[0] = _depth_in[k].at<cv::Vec4w>(i, j)[0];
			//depth.at<cv::Vec3w>(i, j)[1] = _depth_in[k].at<cv::Vec4w>(i, j)[1];
			//depth.at<cv::Vec3w>(i, j)[2] = _depth_in[k].at<cv::Vec4w>(i, j)[2];
			depth.at<unsigned short>(i, j) = _depth_in[k].at<cv::Vec4w>(i, j)[0];

			color.at<cv::Vec3b>(i, j)[0] = _color_in[k].at<cv::Vec4b>(i, j)[0];
			color.at<cv::Vec3b>(i, j)[1] = _color_in[k].at<cv::Vec4b>(i, j)[1];
			color.at<cv::Vec3b>(i, j)[2] = _color_in[k].at<cv::Vec4b>(i, j)[2];
		}
	}

	//cv::imshow("Color image", color);
	/*cv::imshow("Depth image", depth);
	cv::waitKey(100);*/

	char filename[100];
	sprintf_s(filename, "Seq\\KinectV1-7\\RGB_%d.tiff", _idx_thread[k] - 10);
	cv::imwrite(filename, color);
	sprintf_s(filename, "Seq\\KinectV1-7\\Depth_%d.tiff", _idx_thread[k] - 10);
	cv::imwrite(filename, depth);

	depth.release();
	color.release();

	return 0;
}

bool HeadOffV2::Compute3DData() {
	if (_ptsQ.empty())
		return false;

	cl_int ret;
	cl_event evts[3];
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { cDepthWidth, cDepthHeight, 1 };
	ret = clEnqueueWriteImage(_queue[BILATERAL_KER], _depthBuffCL, false, origin, region, cDepthWidth * 4 * sizeof(unsigned short), 0, _depth.front().data, 0, NULL, &evts[0]);
	checkErr(ret, "Unable to write input");
	ret = clEnqueueWriteImage(_queue[VMAP_KER], _SegmentedCL, false, origin, region, cDepthWidth * 4 * sizeof(unsigned char), 0, _segmented_color.front().data, 0, NULL, &evts[1]);
	checkErr(ret, "Unable to write input");

	ret = clEnqueueWriteImage(_queue[VMAP_KER], _RGBMapCL, false, origin, region, cDepthWidth * 4 * sizeof(unsigned char), 0, _color.front().data, 0, NULL, &evts[2]);
	checkErr(ret, "Unable to write input");
	ret = clEnqueueReadImage(_queue[VMAP_KER], _RGBMapCL, false, origin, region, cDepthWidth * 4 * sizeof(unsigned char), 0, _RGBMap.data, 1, &evts[2], NULL);
	checkErr(ret, "Unable to read output");

	cv::Mat tmpLm = _ptsQ.front();
	int nbLM = tmpLm.cols;
	if (nbLM < 43) {
		//cout << "Not enough features" << endl;
		for (int i = 0; i < 43; i++) {
			_landmarks.at<float>(0, i) = 0.0;
			_landmarks.at<float>(1, i) = 0.0;
		}
	}
	else {
		for (int i = 0; i < 43; i++) {
			_landmarks.at<float>(0, i) = tmpLm.at <float>(0, i);
			_landmarks.at<float>(1, i) = tmpLm.at <float>(1, i);
		}
	}
	//_landmarks = _ptsQ.front().clone();

	/*cv::imshow("color", _color.front());
	int key = cv::waitKey(1);*/
	_idx_curr++;

	// Compute Vertex map
	int gws_x = divUp(cDepthHeight, THREAD_SIZE_X);
	int gws_y = divUp(cDepthWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	ret = clEnqueueNDRangeKernel(_queue[BILATERAL_KER], _kernels[BILATERAL_KER], 2, NULL, gws, lws, 1, &evts[0], NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[BILATERAL_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	cl_event evtVMap[3];
	ret = clEnqueueNDRangeKernel(_queue[VMAP_KER], _kernels[VMAP_KER], 2, NULL, gws, lws, 2, evts, &evtVMap[0]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clEnqueueReadImage(_queue[VMAP_KER], _VMapCL, true, origin, region, cDepthWidth * 4 * sizeof(float), 0, _VMap.data, 1, &evtVMap[0], &evtVMap[1]);
	checkErr(ret, "Unable to read output");

	// Compute Normal map
	ret = clEnqueueNDRangeKernel(_queue[NMAP_KER], _kernels[NMAP_KER], 2, NULL, gws, lws, 1, &evtVMap[0], &evtVMap[2]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clEnqueueReadImage(_queue[NMAP_KER], _NMapCL, true, origin, region, cDepthWidth * 4 * sizeof(float), 0, _NMap.data, 2, &evtVMap[1], NULL);
	checkErr(ret, "Unable to read output");

	ret = clFinish(_queue[VMAP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");
	ret = clFinish(_queue[NMAP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	return true;
}

void HeadOffV2::ComputeNormalesDepth() {
	// Compute normals
	MyPoint *currPt;
	float p1[3];
	float p2[3];
	float p3[3];
	float n_p[3];
	float n_p1[3];
	float n_p2[3];
	float n_p3[3];
	float n_p4[3];
	float norm_n;
	for (int i = 1; i < cDepthHeight - 1; i++) {
		for (int j = 1; j < cDepthWidth - 1; j++) {

			if (_depth.front().at<cv::Vec3w>(i, j)[2] == 0)
				continue;

			currPt = _vertices[i*cDepthWidth + j];

			unsigned short n_tot = 0;

			p1[0] = currPt->_x;
			p1[1] = currPt->_y;
			p1[2] = currPt->_z;

			n_p1[0] = 0.0; n_p1[1] = 0.0; n_p1[2] = 0.0;
			n_p2[0] = 0.0; n_p2[1] = 0.0; n_p2[2] = 0.0;
			n_p3[0] = 0.0; n_p3[1] = 0.0; n_p3[2] = 0.0;
			n_p4[0] = 0.0; n_p4[1] = 0.0; n_p4[2] = 0.0;

			////////////////////////// Triangle 1 /////////////////////////////////
			if (_vertices[(i + 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j + 1)] == NULL)
				goto TRIANGLE2;

			p2[0] = _vertices[(i + 1)*cDepthWidth + j]->_x;
			p2[1] = _vertices[(i + 1)*cDepthWidth + j]->_y;
			p2[2] = _vertices[(i + 1)*cDepthWidth + j]->_z;

			p3[0] = _vertices[i*cDepthWidth + (j + 1)]->_x;
			p3[1] = _vertices[i*cDepthWidth + (j + 1)]->_y;
			p3[2] = _vertices[i*cDepthWidth + (j + 1)]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p1[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p1[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p1[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p1[0] * n_p1[0] + n_p1[1] * n_p1[1] + n_p1[2] * n_p1[2]);

				if (norm_n != 0.0) {
					n_p1[0] = n_p1[0] / sqrt(norm_n);
					n_p1[1] = n_p1[1] / sqrt(norm_n);
					n_p1[2] = n_p1[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 2 /////////////////////////////////
		TRIANGLE2:
			if (_vertices[(i - 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j + 1)] == NULL)
				goto TRIANGLE3;

			p2[0] = _vertices[i*cDepthWidth + (j + 1)]->_x;
			p2[1] = _vertices[i*cDepthWidth + (j + 1)]->_y;
			p2[2] = _vertices[i*cDepthWidth + (j + 1)]->_z;

			p3[0] = _vertices[(i - 1)*cDepthWidth + j]->_x;
			p3[1] = _vertices[(i - 1)*cDepthWidth + j]->_y;
			p3[2] = _vertices[(i - 1)*cDepthWidth + j]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p2[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p2[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p2[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p2[0] * n_p2[0] + n_p2[1] * n_p2[1] + n_p2[2] * n_p2[2]);

				if (norm_n != 0.0) {
					n_p2[0] = n_p2[0] / sqrt(norm_n);
					n_p2[1] = n_p2[1] / sqrt(norm_n);
					n_p2[2] = n_p2[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 3 /////////////////////////////////
		TRIANGLE3:
			if (_vertices[(i - 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j - 1)] == NULL)
				goto TRIANGLE4;

			p2[0] = _vertices[(i - 1)*cDepthWidth + j]->_x;
			p2[1] = _vertices[(i - 1)*cDepthWidth + j]->_y;
			p2[2] = _vertices[(i - 1)*cDepthWidth + j]->_z;

			p3[0] = _vertices[i*cDepthWidth + (j - 1)]->_x;
			p3[1] = _vertices[i*cDepthWidth + (j - 1)]->_y;
			p3[2] = _vertices[i*cDepthWidth + (j - 1)]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p3[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p3[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p3[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p3[0] * n_p3[0] + n_p3[1] * n_p3[1] + n_p3[2] * n_p3[2]);

				if (norm_n != 0) {
					n_p3[0] = n_p3[0] / sqrt(norm_n);
					n_p3[1] = n_p3[1] / sqrt(norm_n);
					n_p3[2] = n_p3[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 4 /////////////////////////////////
		TRIANGLE4:
			if (_vertices[(i + 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j - 1)] == NULL)
				goto ENDNORMPROC;

			p2[0] = _vertices[i*cDepthWidth + (j - 1)]->_x;
			p2[1] = _vertices[i*cDepthWidth + (j - 1)]->_y;
			p2[2] = _vertices[i*cDepthWidth + (j - 1)]->_z;

			p3[0] = _vertices[(i + 1)*cDepthWidth + j]->_x;
			p3[1] = _vertices[(i + 1)*cDepthWidth + j]->_y;
			p3[2] = _vertices[(i + 1)*cDepthWidth + j]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p4[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p4[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p4[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p4[0] * n_p4[0] + n_p4[1] * n_p4[1] + n_p4[2] * n_p4[2]);

				if (norm_n != 0) {
					n_p4[0] = n_p4[0] / sqrt(norm_n);
					n_p4[1] = n_p4[1] / sqrt(norm_n);
					n_p4[2] = n_p4[2] / sqrt(norm_n);

					n_tot++;
				}
			}

		ENDNORMPROC:
			if (n_tot == 0) {
				currPt->_Nx = 0.0f;
				currPt->_Ny = 0.0f;
				currPt->_Nz = 0.0f;
				continue;
			}

			n_p[0] = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0]) / float(n_tot);
			n_p[1] = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1]) / float(n_tot);
			n_p[2] = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2]) / float(n_tot);

			norm_n = sqrt(n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);

			if (norm_n != 0) {
				currPt->_Nx = n_p[0] / norm_n;
				currPt->_Ny = n_p[1] / norm_n;
				currPt->_Nz = n_p[2] / norm_n;
			}
			else {
				currPt->_Nx = 0.0f;
				currPt->_Ny = 0.0f;
				currPt->_Nz = 0.0f;
			}

		}
	}
}

void HeadOffV2::DetectFeatures(cv::CascadeClassifier *face_cascade, bool draw){
	float score = 0.f;

	if (_restart) {
		int minFaceH = 50;
		cv::Size minFace(minFaceH, minFaceH); // minimum face size to detect
		vector<cv::Rect> faces;
		face_cascade->detectMultiScale(_imgC, faces, 1.2, 2, 0, minFace);
		if (!faces.empty()) {
			cv::Rect& faceL = *max_element(faces.begin(), faces.end(), compareRect);
			_sdm->detect(_imgC, faceL, _pts, score);
		}
	}
	else {
		_sdm->track(_imgC, _prevPts, _pts, score);
	}

	if (score > _minScore) {
		_restart = false;
		_prevPts = _pts.clone();
	}
	else {
		_restart = true;
	}

	if (!_restart) {
		if (_pts.rows > 0) {
			drawFace(_imgS, _pts);
			/*cv::Mat color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);
			for (int i = 0; i < cDepthHeight; i++) {
				for (int j = 0; j < cDepthWidth; j++) {
					color.at<cv::Vec3b>(i, j)[0] = _imgS.at<cv::Vec4b>(i, j)[0];
					color.at<cv::Vec3b>(i, j)[1] = _imgS.at<cv::Vec4b>(i, j)[1];
					color.at<cv::Vec3b>(i, j)[2] = _imgS.at<cv::Vec4b>(i, j)[2];
				}
			}
			cv::imwrite("FirstRGB.tiff", color);
			int tnpval;
			cin >> tnpval;*/
			//drawPose(_color, _hp.rot);
		}
		//cv::imshow("test", _segmented_color.back());
		//int key = cv::waitKey(1);
	}
}

// Re-scale all blendshapes to match user landmarks
bool HeadOffV2::Rescale(vector<MyMesh *> Blendshape) {
	MyMesh * RefMesh = Blendshape[0];

	if (_landmarks.cols == 0) {
		return false;
	}
	// Compute average factor in X length from outer corner of eyes
	MyPoint *LeftEyeL = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[2]);
	if (LeftEyeL->_x == 0.0f && LeftEyeL->_y == 0.0f && LeftEyeL->_z == 0.0f) {
		cout << "Landmark LeftEyeL NULL" << endl;
		delete LeftEyeL;
		return false;
	}
	MyPoint *RightEyeR = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[2]);
	if (RightEyeR->_x == 0.0f && RightEyeR->_y == 0.0f && RightEyeR->_z == 0.0f) {
		cout << "Landmark RightEyeR NULL" << endl;
		delete LeftEyeL;
		delete RightEyeR;
		return false;
	}
	float eye_dist = sqrt((LeftEyeL->_x - RightEyeR->_x)*(LeftEyeL->_x - RightEyeR->_x) + (LeftEyeL->_y - RightEyeR->_y)*(LeftEyeL->_y - RightEyeR->_y) + (LeftEyeL->_z - RightEyeR->_z)*(LeftEyeL->_z - RightEyeR->_z));
	float eye_dist_mesh = sqrt((RefMesh->Landmark(19)->_x - RefMesh->Landmark(28)->_x)*(RefMesh->Landmark(19)->_x - RefMesh->Landmark(28)->_x) +
		(RefMesh->Landmark(19)->_y - RefMesh->Landmark(28)->_y)*(RefMesh->Landmark(19)->_y - RefMesh->Landmark(28)->_y) +
		(RefMesh->Landmark(19)->_z - RefMesh->Landmark(28)->_z)*(RefMesh->Landmark(19)->_z - RefMesh->Landmark(28)->_z));
	float fact = eye_dist / eye_dist_mesh;
	delete LeftEyeL;
	delete RightEyeR;

	// Compute average factor in X length from inner corner of eyes
	MyPoint *LeftEyeR = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[2]);
	if (LeftEyeR->_x == 0.0f && LeftEyeR->_y == 0.0f && LeftEyeR->_z == 0.0f) {
		cout << "Landmark LeftEyeR NULL" << endl;
		delete LeftEyeR;
		return false;
	}
	MyPoint *RightEyeL = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[2]);
	if (RightEyeL->_x == 0.0f && RightEyeL->_y == 0.0f && RightEyeL->_z == 0.0f) {
		cout << "Landmark RightEyeL NULL" << endl;
		delete LeftEyeR;
		delete RightEyeL;
		return false;
	}
	eye_dist = sqrt((LeftEyeR->_x - RightEyeL->_x)*(LeftEyeR->_x - RightEyeL->_x) + (LeftEyeR->_y - RightEyeL->_y)*(LeftEyeR->_y - RightEyeL->_y) + (LeftEyeR->_z - RightEyeL->_z)*(LeftEyeR->_z - RightEyeL->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(22)->_x - RefMesh->Landmark(25)->_x)*(RefMesh->Landmark(22)->_x - RefMesh->Landmark(25)->_x) +
		(RefMesh->Landmark(22)->_y - RefMesh->Landmark(25)->_y)*(RefMesh->Landmark(22)->_y - RefMesh->Landmark(25)->_y) +
		(RefMesh->Landmark(22)->_z - RefMesh->Landmark(25)->_z)*(RefMesh->Landmark(22)->_z - RefMesh->Landmark(25)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete LeftEyeR;
	delete RightEyeL;

	// Compute average factor in X length from mouth
	MyPoint *LeftMouth = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[2]);
	if (LeftMouth->_x == 0.0f && LeftMouth->_y == 0.0f && LeftMouth->_z == 0.0f) {
		cout << "Landmark LeftMouth NULL" << endl;
		delete LeftMouth;
		return false;
	}
	MyPoint *RightMouth = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[2]);
	if (RightMouth->_x == 0.0f && RightMouth->_y == 0.0f && RightMouth->_z == 0.0f) {
		cout << "Landmark RightEyeL NULL" << endl;
		delete LeftMouth;
		delete RightMouth;
		return false;
	}
	eye_dist = sqrt((LeftMouth->_x - RightMouth->_x)*(LeftMouth->_x - RightMouth->_x) + (LeftMouth->_y - RightMouth->_y)*(LeftMouth->_y - RightMouth->_y) + (LeftMouth->_z - RightMouth->_z)*(LeftMouth->_z - RightMouth->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(31)->_x - RefMesh->Landmark(37)->_x)*(RefMesh->Landmark(31)->_x - RefMesh->Landmark(37)->_x) +
		(RefMesh->Landmark(31)->_y - RefMesh->Landmark(37)->_y)*(RefMesh->Landmark(31)->_y - RefMesh->Landmark(37)->_y) +
		(RefMesh->Landmark(31)->_z - RefMesh->Landmark(37)->_z)*(RefMesh->Landmark(31)->_z - RefMesh->Landmark(37)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete LeftMouth;
	delete RightMouth;


	// Compute average factor in Y length from nose
	MyPoint *UpNose = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[2]);
	if (UpNose->_x == 0.0f && UpNose->_y == 0.0f && UpNose->_z == 0.0f) {
		cout << "Landmark UpNose NULL" << endl;
		delete UpNose;
		return false;
	}
	MyPoint *DownNose = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[2]);
	if (DownNose->_x == 0.0f && DownNose->_y == 0.0f && DownNose->_z == 0.0f) {
		cout << "Landmark DownNose NULL" << endl;
		delete UpNose;
		delete DownNose;
		return false;
	}
	eye_dist = sqrt((UpNose->_x - DownNose->_x)*(UpNose->_x - DownNose->_x) + (UpNose->_y - DownNose->_y)*(UpNose->_y - DownNose->_y) + (UpNose->_z - DownNose->_z)*(UpNose->_z - DownNose->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(10)->_x - RefMesh->Landmark(16)->_x)*(RefMesh->Landmark(10)->_x - RefMesh->Landmark(16)->_x) +
		(RefMesh->Landmark(10)->_y - RefMesh->Landmark(16)->_y)*(RefMesh->Landmark(10)->_y - RefMesh->Landmark(16)->_y) +
		(RefMesh->Landmark(10)->_z - RefMesh->Landmark(16)->_z)*(RefMesh->Landmark(10)->_z - RefMesh->Landmark(16)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete UpNose;
	delete DownNose;

	fact = fact / 4.0;
	cout << "Scale factor: " << fact << endl;

	for (vector<MyMesh *>::iterator it = Blendshape.begin(); it != Blendshape.end(); it++) {
		(*it)->Scale(fact);
	}
	return true;
}

bool HeadOffV2::AlignToFace(vector<MyMesh *> Blendshape, bool inverted) {
	// Rotate the reference mesh and compute translation
	MyMesh * RefMesh = Blendshape[0];
	_hpe->estimateHP(_landmarks, _hp);
	cv::Mat Rotation;
	if (inverted)
		Rotation = _hp.rot.inv();
	else
		Rotation = _hp.rot;

	RefMesh->Rotate(Rotation);

	MyPoint *Nose = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 13))*cDepthWidth + Myround(_landmarks.at<float>(0, 13)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 13))*cDepthWidth + Myround(_landmarks.at<float>(0, 13)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 13))*cDepthWidth + Myround(_landmarks.at<float>(0, 13)))[2]);
	if (Nose->_x == 0.0f && Nose->_y == 0.0f && Nose->_z == 0.0f) {
		cout << "Landmark Nose NULL" << endl;
		delete Nose;
		return false;
	}
	MyPoint *Nose_mesh = RefMesh->Landmark(13);

	cv::Point3f xyz;
	xyz.x = Nose->_x - Nose_mesh->_x;
	xyz.y = Nose->_y - Nose_mesh->_y;
	xyz.z = Nose->_z - Nose_mesh->_z;
	RefMesh->Translate(xyz);
	// Affect values to deformed positions and normals
	RefMesh->AffectToTVal();

	for (vector<MyMesh *>::iterator it = Blendshape.begin() + 1; it != Blendshape.end(); it++) {
		(*it)->Rotate(Rotation);
		(*it)->Translate(xyz);
	}


	// Compute HD Blendshape Vertices 
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	cl_int ret = clEnqueueNDRangeKernel(_queue[DATAPROC_KER], _kernels[DATAPROC_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[DATAPROC_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	delete Nose;
	return true;
}

void HeadOffV2::ElasticRegistrationFull(vector<MyMesh *> Blendshape){

	MyMesh *RefMesh = Blendshape[0];

	/****Compute local tangent plane transforms*****/
	RefMesh->ComputeTgtPlane();

	int iter = 0;
	float pt[3];
	float nmle1[3];
	float nmle2[3];
	float a[3];
	while (iter < 10) {

		/****Point correspondences***/

		/****Solve linear system*****/
		// Build Matrix A
		/*
		A = [-Nx -Ny -Nz 0 ......... 0]
		[0 0 0 -Nx -Ny -Nz 0 ... 0]
		...
		[0............ -Nx -Ny -Nz]
		[0..0...1....-1 .......]
		[0..0...0 1...0 -1 ....]
		[0..0...0 0 1.0 0 -1...]*/

		bool found_coresp;
		float min_dist;
		int p_indx[2];
		int li, ui, lj, uj;
		float DepthP[3];
		float dist;
		float pointClose[3];
		int indx_V = 0;
		int nbMatches = 0;
		int indx_Match = 0;
		for (vector<Point3D<float> *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
			// Search for corresponding point
			continue;
			min_dist = 1000.0;
			if ((*it)->_Nz < 0.0f)
				continue;

			pt[0] = (*it)->_x;// (*it)->_x * _Rotation_inv(0, 0) + (*it)->_y*_Rotation_inv(0, 1) + (*it)->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = (*it)->_y;//(*it)->_x * _Rotation_inv(1, 0) + (*it)->_y*_Rotation_inv(1, 1) + (*it)->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = (*it)->_z;//(*it)->_x * _Rotation_inv(2, 0) + (*it)->_y*_Rotation_inv(2, 1) + (*it)->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

			int uv_u = Myround((*it)->_u * float(BumpHeight));
			int uv_v = Myround((*it)->_v * float(BumpWidth));
			if (_LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] > 100)
				continue;

			/*** Projective association ***/
			// Project the point onto the depth image
			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_intrinsic[0] + _intrinsic[2]))));
			p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_intrinsic[1] + _intrinsic[3]))));

			li = max((cDepthHeight - p_indx[1] - 1) - 2, 0);
			ui = min((cDepthHeight - p_indx[1] - 1) + 3, cDepthHeight);
			lj = max(p_indx[0] - 2, 0);
			uj = min(p_indx[0] + 3, cDepthWidth);

			for (int i = li; i < ui; i++) {
				for (int j = lj; j < uj; j++) {
					DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];
					DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
					DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
					if (DepthP[0] == 0.0 && DepthP[1] == 0.0 && DepthP[2] == 0.0f)
						continue;

					dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));

					if (dist < min_dist) {
						min_dist = dist;
					}
				}
			}

			if (min_dist < 0.05)
				nbMatches++;
		}

		cout << "nbMatches: " << nbMatches << endl;

		int nb_columns = 3 * RefMesh->size();
		int nb_lines = 3 * (43 + nbMatches) + 3 * RefMesh->sizeV();
		SpMat A(nb_lines, nb_columns);
		Eigen::VectorXd b1(nb_lines);
		vector<TrplType> tripletList;

		for (vector<Point3D<float> *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
			// Search for corresponding point
			continue;
			found_coresp = false;
			min_dist = 1000.0;
			if ((*it)->_Nz < 0.0f) {
				indx_V++;
				continue;
			}

			pt[0] = (*it)->_x;// (*it)->_x * _Rotation_inv(0, 0) + (*it)->_y*_Rotation_inv(0, 1) + (*it)->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = (*it)->_y;//(*it)->_x * _Rotation_inv(1, 0) + (*it)->_y*_Rotation_inv(1, 1) + (*it)->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = (*it)->_z;//(*it)->_x * _Rotation_inv(2, 0) + (*it)->_y*_Rotation_inv(2, 1) + (*it)->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

			int uv_u = Myround((*it)->_u * float(BumpHeight));
			int uv_v = Myround((*it)->_v * float(BumpWidth));
			if (_LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] > 100) {
				indx_V++;
				continue;
			}

			/*** Projective association ***/
			// Project the point onto the depth image
			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_intrinsic[0] + _intrinsic[2]))));
			p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_intrinsic[1] + _intrinsic[3]))));

			li = max((cDepthHeight - p_indx[1] - 1) - 2, 0);
			ui = min((cDepthHeight - p_indx[1] - 1) + 3, cDepthHeight);
			lj = max(p_indx[0] - 2, 0);
			uj = min(p_indx[0] + 3, cDepthWidth);

			for (int i = li; i < ui; i++) {
				for (int j = lj; j < uj; j++) {
					DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];
					DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
					DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
					if (DepthP[0] == 0.0 && DepthP[1] == 0.0 && DepthP[2] == 0.0f)
						continue;

					dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));

					if (dist < min_dist) {
						min_dist = dist;
						pointClose[0] = DepthP[0];// DepthP[0] * _Rotation(0, 0) + DepthP[1] * _Rotation(0, 1) + DepthP[2] * _Rotation(0, 2) + _Translation(0);
						pointClose[1] = DepthP[1];//DepthP[0] * _Rotation(1, 0) + DepthP[1] * _Rotation(1, 1) + DepthP[2] * _Rotation(1, 2) + _Translation(1);
						pointClose[2] = DepthP[2];//DepthP[0] * _Rotation(2, 0) + DepthP[1] * _Rotation(2, 1) + DepthP[2] * _Rotation(2, 2) + _Translation(2);
					}
				}
			}

			if (min_dist < 0.05)
				found_coresp = true;

			if (found_coresp) {
				tripletList.push_back(TrplType(3 * indx_Match, 3 * indx_V, 1.0));
				tripletList.push_back(TrplType(3 * indx_Match + 1, 3 * indx_V + 1, 1.0));
				tripletList.push_back(TrplType(3 * indx_Match + 2, 3 * indx_V + 2, 1.0));

				b1(3 * indx_Match) = 1.0*double(pointClose[0]);
				b1(3 * indx_Match + 1) = 1.0*double(pointClose[1]);
				b1(3 * indx_Match + 2) = 1.0*double(pointClose[2]);
				indx_Match++;
			}
			indx_V++;
		}

		/*****************************Add landmarks******************************************/
		float Landmark[3];
		for (int i = 0; i < 43; i++) {
			tripletList.push_back(TrplType(3 * (nbMatches + i), 3 * FACIAL_LANDMARKS[i], 1.0));
			tripletList.push_back(TrplType(3 * (nbMatches + i) + 1, 3 * FACIAL_LANDMARKS[i] + 1, 1.0));
			tripletList.push_back(TrplType(3 * (nbMatches + i) + 2, 3 * FACIAL_LANDMARKS[i] + 2, 1.0));

			Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[0];
			Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[1];
			Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[2];
			if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
				cout << "Landmark NULL!!" << endl;
				return;
			}
			b1(3 * (nbMatches + i)) = 1.0*double(Landmark[0]);
			b1(3 * (nbMatches + i) + 1) = 1.0*double(Landmark[1]);
			b1(3 * (nbMatches + i) + 2) = 1.0*double(Landmark[2]);
		}

		/***************Populate matrix from neighboors of the vertices***************************/
		RefMesh->PopulateMatrix(&tripletList, &b1, 3 * (43 + nbMatches));

		A.setFromTriplets(tripletList.begin(), tripletList.end());

		SpMat MatA(nb_columns, nb_columns);
		Eigen::VectorXd b(nb_columns);
		MatA = A.transpose() * A;
		b = A.transpose() * b1;

		Eigen::SimplicialCholesky<SpMat> chol(MatA);  // performs a Cholesky factorization of A
		Eigen::VectorXd xres = chol.solve(b);    // use the factorization to solve for the given right hand side

		RefMesh->AffectToTVectorT(&xres);

		iter++;
	}

	/***********************Transfer expression deformation******************************/
	Eigen::VectorXd boV(3 * RefMesh->size());
	RefMesh->Deform(&boV);

	if (save_data)
		RefMesh->Write(string(dest_name) + string("\\DeformedMeshes\\Neutral.obj"));

	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {

		Eigen::SimplicialCholesky<SpMat> chol(_MatList1[indxMesh]);  // performs a Cholesky factorization of A
		Eigen::VectorXd b = chol.solve(_MatList2[indxMesh] * boV);

		(*itMesh)->AffectToTVector(&b);

		if (save_data)
			(*itMesh)->Write(string(dest_name) + string("\\DeformedMeshes\\") + to_string(indxMesh) + string(".obj"));

		indxMesh++;
	}


	// Compute HD Blendshape Vertices 
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	cl_int ret = clEnqueueNDRangeKernel(_queue[DATAPROC_KER], _kernels[DATAPROC_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[DATAPROC_KER]);
	checkErr(ret, "ComamndQueue::Finish()");
}

void HeadOffV2::ElasticRegistration(vector<MyMesh *> Blendshape){
	if (_landmarks.cols == 0) {
		return;
	}

	MyMesh *RefMesh = Blendshape[0];
	float Landmark[3];

	/****Compute local tangent plane transforms*****/
	RefMesh->ComputeTgtPlane();

	int iter = 0;
	float pt[3];
	float nmle1[3];
	float nmle2[3];
	float a[3];
	while (iter < 3) {

		/****Point correspondences are the facial landmarks***/

		/****Solve linear system*****/
		// Build Matrix A
		/*
		A = [-Nx -Ny -Nz 0 ......... 0]
		[0 0 0 -Nx --Ny -Nz 0 ... 0]
		...
		[0............ -Nx -Ny -Nz]
		[0..0...1....-1 .......]
		[0..0...0 1...0 -1 ....]
		[0..0...0 0 1.0 0 -1...]*/

		int nb_columns = 3 * RefMesh->size();
		int nb_lines = 3 * 43 + 3 * RefMesh->sizeV();
		SpMat A(nb_lines, nb_columns);
		Eigen::VectorXd b1(nb_lines);
		vector<TrplType> tripletList;

		for (int i = 0; i < 43; i++) {
			/*if (i == 14 || i == 15 || i == 17 || i == 18) {
				tripletList.push_back(TrplType(3 * i, 3 * FACIAL_LANDMARKS[16], 1.0));
				tripletList.push_back(TrplType(3 * i + 1, 3 * FACIAL_LANDMARKS[16] + 1, 1.0));
				tripletList.push_back(TrplType(3 * i + 2, 3 * FACIAL_LANDMARKS[16] + 2, 1.0));

				Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[0];
				Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[1];
				Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[2];
				if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
					cout << "Landmark NULL!!" << endl;
					return;
				}
				b1(3 * i) = double(Landmark[0]);
				b1(3 * i + 1) = double(Landmark[1]);
				b1(3 * i + 2) = double(Landmark[2]);
				continue;
			}*/
			tripletList.push_back(TrplType(3 * i, 3 * FACIAL_LANDMARKS[i], 1.0));
			tripletList.push_back(TrplType(3 * i + 1, 3 * FACIAL_LANDMARKS[i] + 1, 1.0));
			tripletList.push_back(TrplType(3 * i + 2, 3 * FACIAL_LANDMARKS[i] + 2, 1.0));

			Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[0];
			Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[1];
			Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[2];
			if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
				cout << "Landmark NULL!!" << endl;
				return;
			}
			b1(3 * i) = 1.0*double(Landmark[0]);
			b1(3 * i + 1) = 1.0*double(Landmark[1]);
			b1(3 * i + 2) = 1.0*double(Landmark[2]);
		}

		/***************Populate matrix from neighboors of the vertices***************************/
		RefMesh->PopulateMatrix(&tripletList, &b1, 3 * 43);

		A.setFromTriplets(tripletList.begin(), tripletList.end());

		SpMat MatA(nb_columns, nb_columns);
		Eigen::VectorXd b(nb_columns);
		MatA = A.transpose() * A;
		b = A.transpose() * b1;

		Eigen::SimplicialCholesky<SpMat> chol(MatA);  // performs a Cholesky factorization of A
		Eigen::VectorXd xres = chol.solve(b);    // use the factorization to solve for the given right hand side

		RefMesh->AffectToTVectorT(&xres);

		iter++;
	}

	/***********************Transfer expression deformation******************************/
	Eigen::VectorXd boV(3 * RefMesh->size());
	RefMesh->Deform(&boV);

	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {

		Eigen::SimplicialCholesky<SpMat> chol(_MatList1[indxMesh]);  // performs a Cholesky factorization of A
		Eigen::VectorXd b = chol.solve(_MatList2[indxMesh] * boV);

		(*itMesh)->AffectToTVector(&b);

		indxMesh++;
	}


	// Compute HD Blendshape Vertices 
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	cl_int ret = clEnqueueNDRangeKernel(_queue[DATAPROC_KER], _kernels[DATAPROC_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[DATAPROC_KER]);
	checkErr(ret, "ComamndQueue::Finish()");
}

void HeadOffV2::ComputeAffineTransfo(vector<MyMesh *> Blendshape) {
	MyMesh * RefMesh = Blendshape[0];

	// Inititialisation
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		vector<Eigen::Matrix3f> TmpList;
		for (int i = 0; i < Blendshape.size() - 1; i++) {
			Eigen::Matrix3f A;
			TmpList.push_back(A);
		}
		_TransfoExpression.push_back(TmpList);
	}

	vector<SpMat> MatList;
	MyMesh * bi;
	float nmle[3];
	float summit4[3];
	float nmlei[3];
	float summit4i[3];
	Eigen::Matrix3f So;
	Eigen::Matrix3f Si;
	Eigen::Matrix3f Scurr;
	int indxFace = 0;

	Face *CurrFace;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		CurrFace = *it;
		// Compute normal and weight of the face
		GetWeightedNormal(RefMesh, CurrFace, nmle);

		//Compute tetrahedron for b0
		summit4[0] = (RefMesh->_vertices[CurrFace->_v1]->_x + RefMesh->_vertices[CurrFace->_v2]->_x + RefMesh->_vertices[CurrFace->_v3]->_x) / 3.0 + nmle[0];
		summit4[1] = (RefMesh->_vertices[CurrFace->_v1]->_y + RefMesh->_vertices[CurrFace->_v2]->_y + RefMesh->_vertices[CurrFace->_v3]->_y) / 3.0 + nmle[1];
		summit4[2] = (RefMesh->_vertices[CurrFace->_v1]->_z + RefMesh->_vertices[CurrFace->_v2]->_z + RefMesh->_vertices[CurrFace->_v3]->_z) / 3.0 + nmle[2];

		So(0, 0) = (RefMesh->_vertices[CurrFace->_v2]->_x - RefMesh->_vertices[CurrFace->_v1]->_x);	 So(0, 1) = (RefMesh->_vertices[CurrFace->_v3]->_x - RefMesh->_vertices[CurrFace->_v1]->_x);		So(0, 2) = (summit4[0] - RefMesh->_vertices[CurrFace->_v1]->_x);
		So(1, 0) = (RefMesh->_vertices[CurrFace->_v2]->_y - RefMesh->_vertices[CurrFace->_v1]->_y);	 So(1, 1) = (RefMesh->_vertices[CurrFace->_v3]->_y - RefMesh->_vertices[CurrFace->_v1]->_y);		So(1, 2) = (summit4[1] - RefMesh->_vertices[CurrFace->_v1]->_y);
		So(2, 0) = (RefMesh->_vertices[CurrFace->_v2]->_z - RefMesh->_vertices[CurrFace->_v1]->_z);	 So(2, 1) = (RefMesh->_vertices[CurrFace->_v3]->_z - RefMesh->_vertices[CurrFace->_v1]->_z);		So(2, 2) = (summit4[2] - RefMesh->_vertices[CurrFace->_v1]->_z);

		// Go through all other blendshapes
		int indxMesh = 0;
		for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {
			bi = (*itMesh);
			// Compute normal and weight of the face
			GetWeightedNormal(bi, CurrFace, nmlei);

			//Compute tetrahedron for b0
			summit4i[0] = (bi->_vertices[CurrFace->_v1]->_x + bi->_vertices[CurrFace->_v2]->_x + bi->_vertices[CurrFace->_v3]->_x) / 3.0 + nmlei[0];
			summit4i[1] = (bi->_vertices[CurrFace->_v1]->_y + bi->_vertices[CurrFace->_v2]->_y + bi->_vertices[CurrFace->_v3]->_y) / 3.0 + nmlei[1];
			summit4i[2] = (bi->_vertices[CurrFace->_v1]->_z + bi->_vertices[CurrFace->_v2]->_z + bi->_vertices[CurrFace->_v3]->_z) / 3.0 + nmlei[2];

			Si(0, 0) = (bi->_vertices[CurrFace->_v2]->_x - bi->_vertices[CurrFace->_v1]->_x);	 Si(0, 1) = (bi->_vertices[CurrFace->_v3]->_x - bi->_vertices[CurrFace->_v1]->_x);		Si(0, 2) = (summit4i[0] - bi->_vertices[CurrFace->_v1]->_x);
			Si(1, 0) = (bi->_vertices[CurrFace->_v2]->_y - bi->_vertices[CurrFace->_v1]->_y);	 Si(1, 1) = (bi->_vertices[CurrFace->_v3]->_y - bi->_vertices[CurrFace->_v1]->_y);		Si(1, 2) = (summit4i[1] - bi->_vertices[CurrFace->_v1]->_y);
			Si(2, 0) = (bi->_vertices[CurrFace->_v2]->_z - bi->_vertices[CurrFace->_v1]->_z);	 Si(2, 1) = (bi->_vertices[CurrFace->_v3]->_z - bi->_vertices[CurrFace->_v1]->_z);		Si(2, 2) = (summit4i[2] - bi->_vertices[CurrFace->_v1]->_z);

			_TransfoExpression[indxFace][indxMesh] = Si * So.inverse();

			indxMesh++;
		}
		indxFace++;
	}

	// Compute Matrix F that fix points on the backface.

	float nu = 100.0;
	SpMat F(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	vector<TrplType> tripletListF;
	int indx = 0;
	for (vector<MyPoint *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
		if ((*it)->_BackPoint) {
			tripletListF.push_back(TrplType(3 * indx, 3 * indx, 1.0));
			tripletListF.push_back(TrplType(3 * indx + 1, 3 * indx + 1, 1.0));
			tripletListF.push_back(TrplType(3 * indx + 2, 3 * indx + 2, 1.0));
		}

		indx++;
	}
	F.setFromTriplets(tripletListF.begin(), tripletListF.end());

	// Compute Matrix G that transform vertices to edges.
	SpMat G(6 * RefMesh->_triangles.size(), 3 * RefMesh->_vertices.size());
	vector<TrplType> tripletList;
	indx = 0;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		CurrFace = *it;
		tripletList.push_back(TrplType(6 * indx, 3 * CurrFace->_v1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 1, 3 * CurrFace->_v1 + 1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 2, 3 * CurrFace->_v1 + 2, -1.0));
		tripletList.push_back(TrplType(6 * indx + 3, 3 * CurrFace->_v1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 4, 3 * CurrFace->_v1 + 1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 5, 3 * CurrFace->_v1 + 2, -1.0));


		tripletList.push_back(TrplType(6 * indx, 3 * CurrFace->_v2, 1.0));
		tripletList.push_back(TrplType(6 * indx + 1, 3 * CurrFace->_v2 + 1, 1.0));
		tripletList.push_back(TrplType(6 * indx + 2, 3 * CurrFace->_v2 + 2, 1.0));
		tripletList.push_back(TrplType(6 * indx + 3, 3 * CurrFace->_v3, 1.0));
		tripletList.push_back(TrplType(6 * indx + 4, 3 * CurrFace->_v3 + 1, 1.0));
		tripletList.push_back(TrplType(6 * indx + 5, 3 * CurrFace->_v3 + 2, 1.0));

		indx++;
	}

	G.setFromTriplets(tripletList.begin(), tripletList.end());

	SpMat MatT(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	SpMat MatTmp(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	Eigen::VectorXd b(3 * RefMesh->_vertices.size());
	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {
		//Compute the affine transfo matrix
		SpMat H(6 * RefMesh->_triangles.size(), 6 * RefMesh->_triangles.size());
		vector<TrplType> tripletListH;
		indx = 0;
		for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
			tripletListH.push_back(TrplType(6 * indx, 6 * indx, _TransfoExpression[indx][indxMesh](0, 0)));		tripletListH.push_back(TrplType(6 * indx, 6 * indx + 1, _TransfoExpression[indx][indxMesh](0, 1)));		tripletListH.push_back(TrplType(6 * indx, 6 * indx + 2, _TransfoExpression[indx][indxMesh](0, 2)));
			tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx, _TransfoExpression[indx][indxMesh](1, 0)));	tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx + 1, _TransfoExpression[indx][indxMesh](1, 1)));	tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx + 2, _TransfoExpression[indx][indxMesh](1, 2)));
			tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx, _TransfoExpression[indx][indxMesh](2, 0)));	tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx + 1, _TransfoExpression[indx][indxMesh](2, 1)));	tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx + 2, _TransfoExpression[indx][indxMesh](2, 2)));

			tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 3, _TransfoExpression[indx][indxMesh](0, 0)));	tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 4, _TransfoExpression[indx][indxMesh](0, 1)));	tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 5, _TransfoExpression[indx][indxMesh](0, 2)));
			tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 3, _TransfoExpression[indx][indxMesh](1, 0)));	tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 4, _TransfoExpression[indx][indxMesh](1, 1)));	tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 5, _TransfoExpression[indx][indxMesh](1, 2)));
			tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 3, _TransfoExpression[indx][indxMesh](2, 0)));	tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 4, _TransfoExpression[indx][indxMesh](2, 1)));	tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 5, _TransfoExpression[indx][indxMesh](2, 2)));

			indx++;
		}

		H.setFromTriplets(tripletListH.begin(), tripletListH.end());

		MatTmp = (G.transpose()*G + nu*F);
		MatT = (G.transpose()*H*G + nu*F);
		_MatList1.push_back(MatTmp);
		_MatList2.push_back(MatT);
		indxMesh++;
	}

}

void HeadOffV2::GenerateBump(vector<MyMesh *> Blendshape, int x, int y, int width, int height) {
	/*** Obj: Compute next state for the Bump image w.r.t current frame ***/
	MyMesh *RefMesh = Blendshape[0];

	/*** For each pixel of the Bump image update its value ***/
	float pt[3];
	float thresh = 0.01; // 2 cm

	// Build graph
	float pos[3];
	float u, v; // texture coordinates
	float weights[4]; // local coordinates w.r.t summit of the face
	float v1[3];
	float v2[3];
	float nmle[3];

	float nmleTmp[3][3];
	float ptTmp[3][3];

	unsigned char flag = 0;
	float d;
	float sum_tot = 0.0;
	float count = 0.0;
	Face *CurrFace;
	for (int i = y; i < y + height; i++) {
		for (int j = x; j < x + width; j++) {

			if (_Bump.at<cv::Vec4f>(i, j)[2] == 0)
				continue;

			CurrFace = RefMesh->_triangles[int(_Bump.at<cv::Vec4f>(i, j)[2])];
			weights[0] = _WeightMap.at<cv::Vec4f>(i, j)[0];
			weights[1] = _WeightMap.at<cv::Vec4f>(i, j)[1];
			weights[2] = _WeightMap.at<cv::Vec4f>(i, j)[2];
			if (weights[0] == -1.0)
				continue;

			flag = 0;
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 0))
				flag = 1; // Left eye up
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 0))
				flag = 2; // Left eye down
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 255))
				flag = 3; // right eye up
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 255))
				flag = 4; // right eye down
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 0))
				flag = 5; // mouth up
			if ((_LabelsMask.at<cv::Vec3b>(i, j)[2] == 0) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[1] == 255) &&
				(_LabelsMask.at<cv::Vec3b>(i, j)[0] == 255))
				flag = 6; // mouth down

			for (int k = 0; k < 3; k++) {
				nmle[k] = 0.0;
				pt[k] = 0.0;
			}
			pt[0] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_x + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_x + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_x) / (weights[0] + weights[1] + weights[2]);
			pt[1] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_y + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_y + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_y) / (weights[0] + weights[1] + weights[2]);
			pt[2] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_z + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_z + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_z) / (weights[0] + weights[1] + weights[2]);
			float ptRef[3];
			ptRef[0] = pt[0]; ptRef[1] = pt[1]; ptRef[2] = pt[2];

			nmle[0] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_Nx + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_Nx + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_Nx) / (weights[0] + weights[1] + weights[2]);
			nmle[1] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_Ny + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_Ny + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_Ny) / (weights[0] + weights[1] + weights[2]);
			nmle[2] = (weights[0] * RefMesh->_vertices[CurrFace->_v1]->_Nz + weights[1] * RefMesh->_vertices[CurrFace->_v2]->_Nz + weights[2] * RefMesh->_vertices[CurrFace->_v3]->_Nz) / (weights[0] + weights[1] + weights[2]);
			normalize(nmle);
			float nmleRef[3];
			nmleRef[0] = nmle[0]; nmleRef[1] = nmle[1]; nmleRef[2] = nmle[2];

			float nTmp[3];
			float pTmp[3];
			for (int k = 1; k < 49; k++) {
				nTmp[0] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_Nx + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_Nx + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_Nx) / (weights[0] + weights[1] + weights[2]);
				nTmp[1] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_Ny + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_Ny + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_Ny) / (weights[0] + weights[1] + weights[2]);
				nTmp[2] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_Nz + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_Nz + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_Nz) / (weights[0] + weights[1] + weights[2]);
				normalize(nTmp);

				pTmp[0] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_x + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_x + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_x) / (weights[0] + weights[1] + weights[2]);
				pTmp[1] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_y + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_y + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_y) / (weights[0] + weights[1] + weights[2]);
				pTmp[2] = (weights[0] * Blendshape[k]->_vertices[CurrFace->_v1]->_z + weights[1] * Blendshape[k]->_vertices[CurrFace->_v2]->_z + weights[2] * Blendshape[k]->_vertices[CurrFace->_v3]->_z) / (weights[0] + weights[1] + weights[2]);

				// This blended normal is not really a normal since it may be not normalized
				nmle[0] = nmle[0] + (nTmp[0] - nmleRef[0]) * _BlendshapeCoeff[k];
				nmle[1] = nmle[1] + (nTmp[1] - nmleRef[1]) * _BlendshapeCoeff[k];
				nmle[2] = nmle[2] + (nTmp[2] - nmleRef[2]) * _BlendshapeCoeff[k];

				pt[0] = pt[0] + (pTmp[0] - ptRef[0]) * _BlendshapeCoeff[k];
				pt[1] = pt[1] + (pTmp[1] - ptRef[1]) * _BlendshapeCoeff[k];
				pt[2] = pt[2] + (pTmp[2] - ptRef[2]) * _BlendshapeCoeff[k];
			}

			int p_indx[2];
			Point3DGPU *V1 = &_verticesBump[i*BumpWidth + j];

			V1->_x = pt[0]; V1->_y = pt[1]; V1->_z = pt[2];
			V1->_Nx = nmle[0]; V1->_Ny = nmle[1]; V1->_Nz = nmle[2];

			pt[0] = V1->_x * _Rotation_inv(0, 0) + V1->_y * _Rotation_inv(0, 1) + V1->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = V1->_x * _Rotation_inv(1, 0) + V1->_y * _Rotation_inv(1, 1) + V1->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = V1->_x * _Rotation_inv(2, 0) + V1->_y * _Rotation_inv(2, 1) + V1->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

			nmle[0] = V1->_Nx * _Rotation_inv(0, 0) + V1->_Ny * _Rotation_inv(0, 1) + V1->_Nz * _Rotation_inv(0, 2);
			nmle[1] = V1->_Nx * _Rotation_inv(1, 0) + V1->_Ny * _Rotation_inv(1, 1) + V1->_Nz * _Rotation_inv(1, 2);
			nmle[2] = V1->_Nx * _Rotation_inv(2, 0) + V1->_Ny * _Rotation_inv(2, 1) + V1->_Nz * _Rotation_inv(2, 2);

			float new_bump, weight, fact_curr, best_state, min_dist, bum_val = 0;
			if (nmle[2] < 0.0) {
				goto UPDATE_SHAPE;
			}

			bum_val = _Bump.at<cv::Vec4f>(i, j)[0];
			min_dist = 1.0e10;
			best_state = -1.0;
			fact_curr = Myround(_Bump.at<cv::Vec4f>(i, j)[1]) == 0 ? 1.0 : min(2.0f, sqrt(float(Myround(_Bump.at<cv::Vec4f>(i, j)[1]))));
			for (int sbp = 0; sbp < MESSAGE_LENGTH; sbp++) {
				d = (bum_val + float(sbp - MESSAGE_LENGTH / 2) / fact_curr) / fact_BP;

				pos[0] = pt[0] + d*nmle[0];
				pos[1] = pt[1] + d*nmle[1];
				pos[2] = pt[2] + d*nmle[2];

				// Project the point onto the depth image
				p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pos[0] / fabs(pos[2]))*_intrinsic[0] + _intrinsic[2]))));
				p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pos[1] / fabs(pos[2]))*_intrinsic[1] + _intrinsic[3]))));

				int lb_i = max((cDepthHeight - p_indx[1] - 1) - 0, 0);
				int ub_i = min((cDepthHeight - p_indx[1] - 1) + 1, cDepthHeight);

				int lb_j = max(p_indx[0] - 0, 0);
				int ub_j = min(p_indx[0] + 1, cDepthWidth);

				for (int k = lb_i; k < ub_i; k++) {
					for (int l = lb_j; l < ub_j; l++) {


						if (k < _pRect.y || k > _pRect.y + _pRect.height - 1 || l < _pRect.x || l > _pRect.x + _pRect.width - 1)
							continue;

						float v2[3];
						v2[0] = _VMap.at<cv::Vec3f>(k, l)[0];
						v2[1] = _VMap.at<cv::Vec3f>(k, l)[1];
						v2[2] = _VMap.at<cv::Vec3f>(k, l)[2];
						float n2[3];
						n2[0] = _NMap.at<cv::Vec3f>(k, l)[0];
						n2[1] = _NMap.at<cv::Vec3f>(k, l)[1];
						n2[2] = _NMap.at<cv::Vec3f>(k, l)[2];

						if (n2[0] == 0.0 && n2[1] == 0.0 && n2[2] == 0.0) // !!
							continue;

						//compute distance of point to the normal
						float u_vect[3];
						u_vect[0] = v2[0] - pt[0];
						u_vect[1] = v2[1] - pt[1];
						u_vect[2] = v2[2] - pt[2];

						float proj = u_vect[0] * nmle[0] + u_vect[1] * nmle[1] + u_vect[2] * nmle[2];
						float v_vect[3];
						v_vect[0] = u_vect[0] - proj * nmle[0];
						v_vect[1] = u_vect[1] - proj * nmle[1];
						v_vect[2] = u_vect[2] - proj * nmle[2];
						float dist = sqrt((v2[0] - pos[0]) * (v2[0] - pos[0]) + (v2[1] - pos[1]) * (v2[1] - pos[1]) + (v2[2] - pos[2]) * (v2[2] - pos[2]));
						float dist_to_nmle = sqrt(v_vect[0] * v_vect[0] + v_vect[1] * v_vect[1] + v_vect[2] * v_vect[2]);
						float dist_angle = nmle[0] * n2[0] + nmle[1] * n2[1] + nmle[2] * n2[2];
						bool valid = (flag == 0) || (flag == unsigned char(_VMap.at<cv::Vec4f>(k, l)[3]));

						if (dist_to_nmle < min_dist && dist_angle > 0.6 && valid && dist < 0.005) {
							min_dist = dist_to_nmle;
							best_state = proj * fact_BP;
						}
					}
				}
			}

			if (best_state == -1.0 || min_dist > 0.005) {
				goto UPDATE_SHAPE;
			}

			weight = _Bump.at<cv::Vec4f>(i, j)[1] == 0.0 ? 0.1 : V1->_TNx * _Rotation_inv(2, 0) + V1->_TNy * _Rotation_inv(2, 1) + V1->_TNz * _Rotation_inv(2, 2);
			weight = weight*weight;
			new_bump = (weight*best_state + bum_val*_Bump.at<cv::Vec4f>(i, j)[1]) / (_Bump.at<cv::Vec4f>(i, j)[1] + weight);
			if (_Bump.at<cv::Vec4f>(i, j)[1]< 100.0) {
				_Bump.at<cv::Vec4f>(i, j)[0] = new_bump;
			}

			//Get color
			float p1[3];
			d = _Bump.at<cv::Vec4f>(i, j)[0] / fact_BP;
			p1[0] = pt[0] + d*nmle[0];
			p1[1] = pt[1] + d*nmle[1];
			p1[2] = pt[2] + d*nmle[2];

			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((p1[0] / fabs(p1[2]))*_intrinsic[0] + _intrinsic[2]))));
			p_indx[1] = cDepthHeight - 1 - min(cDepthHeight - 1, max(0, int(round((p1[1] / fabs(p1[2]))*_intrinsic[1] + _intrinsic[3]))));
			if (_vertices[p_indx[1] * cDepthWidth + p_indx[0]] != NULL && _Bump.at<cv::Vec4f>(i, j)[1] < 100.0) {
				_verticesBump[i*BumpWidth + j]._R = (weight*float(_RGBMap.at<cv::Vec3b>(p_indx[1], p_indx[0])[2]) + V1->_R*_Bump.at<cv::Vec4f>(i, j)[1]) / (_Bump.at<cv::Vec4f>(i, j)[1] + weight);  //_vertices[p_indx[1] * cDepthWidth + p_indx[0]]->_R;
				_verticesBump[i*BumpWidth + j]._G = (weight*float(_RGBMap.at<cv::Vec3b>(p_indx[1], p_indx[0])[1]) + V1->_G*_Bump.at<cv::Vec4f>(i, j)[1]) / (_Bump.at<cv::Vec4f>(i, j)[1] + weight);  //_vertices[p_indx[1] * cDepthWidth + p_indx[0]]->_G;
				_verticesBump[i*BumpWidth + j]._B = (weight*float(_RGBMap.at<cv::Vec3b>(p_indx[1], p_indx[0])[0]) + V1->_B*_Bump.at<cv::Vec4f>(i, j)[1]) / (_Bump.at<cv::Vec4f>(i, j)[1] + weight);  //_vertices[p_indx[1] * cDepthWidth + p_indx[0]]->_B;
			}

			if (_Bump.at<cv::Vec4f>(i, j)[1] < 100.0) {
				_Bump.at<cv::Vec4f>(i, j)[1] = _Bump.at<cv::Vec4f>(i, j)[1] + weight;
			}

		UPDATE_SHAPE:
			if (_Bump.at<cv::Vec4f>(i, j)[1] > 0.0) {
				d = _Bump.at<cv::Vec4f>(i, j)[0] / fact_BP;
				_verticesBump[i*BumpWidth + j]._Tx = V1->_x + d*V1->_Nx;
				_verticesBump[i*BumpWidth + j]._Ty = V1->_y + d*V1->_Ny;
				_verticesBump[i*BumpWidth + j]._Tz = V1->_z + d*V1->_Nz;

				/*if (flag == 1) {
				V1->_R = 1.0;
				V1->_G = 0.0;
				V1->_B = 0.0;
				}
				if (flag == 2) {
				V1->_R = 0.0;
				V1->_G = 1.0;
				V1->_B = 0.0;
				}
				if (flag == 3) {
				V1->_R = 0.0;
				V1->_G = 0.0;
				V1->_B = 1.0;
				}
				if (flag == 4) {
				V1->_R = 1.0;
				V1->_G = 0.0;
				V1->_B = 1.0;
				}
				if (flag == 5) {
				V1->_R = 1.0;
				V1->_G = 1.0;
				V1->_B = 0.0;
				}
				if (flag == 6) {
				V1->_R = 0.0;
				V1->_G = 1.0;
				V1->_B = 1.0;
				}*/
			}
		}
	}

	ComputeNormales(x, y, width, height);
}

void HeadOffV2::GenerateBumpGPU(vector<MyMesh *> Blendshape, int x, int y, int width, int height) {
	//cout << "Start Bump " << _TranslationWindow.size() << endl;

	//if (_idx_curr > 20)
	//	return; 

	/// Check stability state
	int stable = 1;
	//int nbT = _TranslationWindow.size();
	//for (int i = 0; i < nbT; i++) {
	//	Eigen::Matrix3f RotationRef = _RotationWindow[i].inverse();
	//	Eigen::Vector3f TransfoRef = -RotationRef * _TranslationWindow[i];
	//	for (int j = 0; j < nbT; j++) {
	//		Eigen::Matrix3f Rinc = RotationRef * _RotationWindow[j];
	//		Eigen::Vector3f tinc = TransfoRef + _TranslationWindow[j];
	//		if ((Rinc - Eigen::Matrix3f::Identity()).norm() > 1.0e-1 && tinc.norm() > 1.0e-1) {
	//			stable = 0;
	//			break;
	//		}
	//	}
	//	if (stable == 0)
	//		break;
	//}
	////if (stable == 0)
	////	cout << "Non stable state from transformation" << endl;

	//if (stable == 1) {
	//	for (int i = 0; i < NB_BS; i++) {
	//		for (int j = 0; j < nbT; j++) {
	//			for (int k = 0; k < nbT; k++) {
	//				if (fabs(_BSCoeff[i][k] - _BSCoeff[i][j]) > 1.0e-1) {
	//					//cout << fabs(_BSCoeff[i][k] - _BSCoeff[i][j]) << endl;
	//					stable = 0;
	//					break;
	//				}
	//			}
	//			if (stable == 0)
	//				break;
	//		}
	//		if (stable == 0)
	//			break;
	//	}
	//	//if (stable == 0)
	//	//	cout << "Non stable state from coefficients" << endl;
	//}

	/*if (stable == 1)
		cout << "Stable state, going to update bump data" << endl;
	else
		cout << "Non stable state, will only add new points" << endl;*/

	stable = 1;
	/*** Obj: Compute next state for the Bump image w.r.t current frame ***/
	cl_event evts[3];
	cl_int ret;
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };

	ret = clSetKernelArg(_kernels[BUMP_KER], 18, sizeof(int), &stable);

	// Compute Vertex map
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	ret = clEnqueueNDRangeKernel(_queue[BUMP_KER], _kernels[BUMP_KER], 2, NULL, gws, lws, 0, NULL, &evts[0]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[BUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueNDRangeKernel(_queue[MEDIANFILTER_KER], _kernels[MEDIANFILTER_KER], 2, NULL, gws, lws, 1, &evts[0], &evts[1]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[MEDIANFILTER_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[BUMP_KER], _BumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _Bump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueReadImage(_queue[BUMP_KER], _VMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _VMapBump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueReadImage(_queue[BUMP_KER], _RGBMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _RGBMapBump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");

	if (save_data) {
		char filename[100];
		cv::Mat color = cv::Mat(BumpHeight, BumpWidth, CV_8UC3);
		for (int i = 0; i < BumpHeight; i++) {
			for (int j = 0; j < BumpWidth; j++) {
				if (_RGBMapBump.at<cv::Vec4f>(i, j)[2] == 0.0 && _RGBMapBump.at<cv::Vec4f>(i, j)[1] == 0.0 && _RGBMapBump.at<cv::Vec4f>(i, j)[0] == 0.0) {
					color.at<cv::Vec3b>(i, j)[0] = 255;
					color.at<cv::Vec3b>(i, j)[1] = 255;
					color.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					color.at<cv::Vec3b>(i, j)[0] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[2]);
					color.at<cv::Vec3b>(i, j)[1] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[1]);
					color.at<cv::Vec3b>(i, j)[2] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[0]);
				}
			}
		}
		//cv::imwrite("FirstRGB.tiff", color);
		sprintf_s(filename, "%s\\Bump\\RGB%d.png", dest_name, _idx);
		cv::imwrite(filename, color);

		for (int i = 0; i < BumpHeight; i++) {
			for (int j = 0; j < BumpWidth; j++) {
				if (_Bump.at<cv::Vec4f>(i, j)[1] == 0.0) {
					color.at<cv::Vec3b>(i, j)[0] = 255;
					color.at<cv::Vec3b>(i, j)[1] = 255;
					color.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					int pixelValue = unsigned char(((3000.0f + 50.0f*_Bump.at<cv::Vec4f>(i, j)[0]) / 6000.0f)*255.0f);
					//int pixelValue = unsigned char(((1000.0f + 50.0f*_Bump.at<cv::Vec4f>(i, j)[0]) / 2000.0f)*255.0f);
					if (pixelValue < 128) {
						color.at<cv::Vec3b>(i, j)[2] = 0;
						color.at<cv::Vec3b>(i, j)[1] = 0 + 2 * pixelValue;
						color.at<cv::Vec3b>(i, j)[0] = 255 - 2 * pixelValue;
					}
					else {
						color.at<cv::Vec3b>(i, j)[2] = 0 + 2 * (pixelValue - 128);
						color.at<cv::Vec3b>(i, j)[1] = 255 - 2 * (pixelValue - 128);
						color.at<cv::Vec3b>(i, j)[0] = 0;
					}
				}
			}
		}
		sprintf_s(filename, "%s\\Bump\\Bump%d.png", dest_name, _idx);
		cv::imwrite(filename, color);
	}

	//ret = clEnqueueCopyImage(_queue[BUMP_KER], _BumpSwapCL, _BumpCL, origin, origin, region, 1, &evts[1], NULL);
	//checkErr(ret, "Unable to copy output");
	//ret = clEnqueueCopyImage(_queue[BUMP_KER], _RGBMapBumpSwapCL, _RGBMapBumpCL, origin, origin, region, 1, &evts[1], NULL);
	//checkErr(ret, "Unable to copy output");


	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 1, &evts[1], &evts[2]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[NMAPBUMP_KER], _NMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _NMapBump.data, 1, &evts[2], NULL);
	checkErr(ret, "Unable to read output");

	//cout << "End Bump" << endl;

	//cv::imshow("Nmap", _NMapBump);
	//cv::waitKey(1);
}

void HeadOffV2::GenerateBumpVMPGPU(vector<MyMesh *> Blendshape, int x, int y, int width, int height) {

	/// Check stability state
	int stable = 1;

	/*** Obj: Compute next state for the Bump image w.r.t current frame ***/
	cl_event evts[3];
	cl_int ret;
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };

	int D_gws_x = divUp(cDepthHeight, THREAD_SIZE_X);
	int D_gws_y = divUp(cDepthWidth, THREAD_SIZE_Y);
	size_t D_gws[2] = { D_gws_x*THREAD_SIZE_X, D_gws_y*THREAD_SIZE_Y };
	size_t D_lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	//cout << "Depth: " << cDepthHeight << ",  " << cDepthWidth << endl;

	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	//cout << "Bump: " << BumpHeight << ",  " << BumpWidth << endl;

	/////////////////////////////////////////////////////////////////////////////////
	//////////////////////Compute current GRAPH connectivity/////////////////////////
	/////////////////////////////////////////////////////////////////////////////////
	ret = clEnqueueNDRangeKernel(_queue[INITVMP_KER], _kernels[INITVMP_KER], 2, NULL, D_gws, D_lws, 0, NULL, &evts[0]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[INITVMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueNDRangeKernel(_queue[GRAPH_KER], _kernels[GRAPH_KER], 2, NULL, gws, lws, 1, &evts[0], &evts[2]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[GRAPH_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	/////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////Pass messages////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////
	ret = clEnqueueNDRangeKernel(_queue[VMP_KER], _kernels[VMP_KER], 2, NULL, gws, lws, 1, &evts[2], &evts[0]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[VMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueNDRangeKernel(_queue[MEDIANFILTER_KER], _kernels[MEDIANFILTER_KER], 2, NULL, gws, lws, 1, &evts[0], &evts[1]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[MEDIANFILTER_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[VMP_KER], _BumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _Bump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueReadImage(_queue[VMP_KER], _VMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _VMapBump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueReadImage(_queue[VMP_KER], _RGBMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _RGBMapBump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read output");

	if (save_data) {
		char filename[100];
		cv::Mat color = cv::Mat(BumpHeight, BumpWidth, CV_8UC3);
		for (int i = 0; i < BumpHeight; i++) {
			for (int j = 0; j < BumpWidth; j++) {
				if (_RGBMapBump.at<cv::Vec4f>(i, j)[2] == 0.0 && _RGBMapBump.at<cv::Vec4f>(i, j)[1] == 0.0 && _RGBMapBump.at<cv::Vec4f>(i, j)[0] == 0.0) {
					color.at<cv::Vec3b>(i, j)[0] = 255;
					color.at<cv::Vec3b>(i, j)[1] = 255;
					color.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					color.at<cv::Vec3b>(i, j)[0] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[2]);
					color.at<cv::Vec3b>(i, j)[1] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[1]);
					color.at<cv::Vec3b>(i, j)[2] = unsigned char(_RGBMapBump.at<cv::Vec4f>(i, j)[0]);
				}
			}
		}
		//cv::imwrite("FirstRGB.tiff", color);
		sprintf_s(filename, "%s\\Bump\\RGB%d.png", dest_name, _idx);
		cv::imwrite(filename, color);

		for (int i = 0; i < BumpHeight; i++) {
			for (int j = 0; j < BumpWidth; j++) {
				if (_Bump.at<cv::Vec4f>(i, j)[1] == 0.0) {
					color.at<cv::Vec3b>(i, j)[0] = 255;
					color.at<cv::Vec3b>(i, j)[1] = 255;
					color.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					int pixelValue = unsigned char(((3000.0f + 50.0f*_Bump.at<cv::Vec4f>(i, j)[0]) / 6000.0f)*255.0f);
					//int pixelValue = unsigned char(((1000.0f + 50.0f*_Bump.at<cv::Vec4f>(i, j)[0]) / 2000.0f)*255.0f);
					if (pixelValue < 128) {
						color.at<cv::Vec3b>(i, j)[2] = 0;
						color.at<cv::Vec3b>(i, j)[1] = 0 + 2 * pixelValue;
						color.at<cv::Vec3b>(i, j)[0] = 255 - 2 * pixelValue;
					}
					else {
						color.at<cv::Vec3b>(i, j)[2] = 0 + 2 * (pixelValue - 128);
						color.at<cv::Vec3b>(i, j)[1] = 255 - 2 * (pixelValue - 128);
						color.at<cv::Vec3b>(i, j)[0] = 0;
					}
				}
			}
		}
		sprintf_s(filename, "%s\\Bump\\Bump%d.png", dest_name, _idx);
		cv::imwrite(filename, color);
	}

	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 1, &evts[1], &evts[2]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[NMAPBUMP_KER], _NMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _NMapBump.data, 1, &evts[2], NULL);
	checkErr(ret, "Unable to read output");
}

void HeadOffV2::ComputeNormales(int x, int y, int width, int height) {
	//	// Compute normals
	//	Point3DGPU *currVBP;
	//	float p1[3];
	//	float p2[3];
	//	float p3[3];
	//	float n_p[3];
	//	float n_p1[3];
	//	float n_p2[3];
	//	float n_p3[3];
	//	float n_p4[3];
	//	float norm_n;
	//	int sx = max(1, x);
	//	int sy = max(1, y);
	//	int uy = min(BumpHeight - 1, y+height);
	//	int ux = min(BumpWidth - 1, x+width);
	//	for (int i = sy; i < uy - 1; i++) {
	//		for (int j = sx; j < ux - 1; j++) {
	//			if (_Bump[4*(i*BumpWidth + j)+1] == 0.0)
	//				continue;
	//
	//			currVBP = &_verticesBump[i*BumpWidth + j];
	//
	//			unsigned short n_tot = 0;
	//
	//			p1[0] = currVBP->_Tx;
	//			p1[1] = currVBP->_Ty;
	//			p1[2] = currVBP->_Tz;
	//
	//			n_p1[0] = 0.0; n_p1[1] = 0.0; n_p1[2] = 0.0;
	//			n_p2[0] = 0.0; n_p2[1] = 0.0; n_p2[2] = 0.0;
	//			n_p3[0] = 0.0; n_p3[1] = 0.0; n_p3[2] = 0.0;
	//			n_p4[0] = 0.0; n_p4[1] = 0.0; n_p4[2] = 0.0;
	//
	//			////////////////////////// Triangle 1 /////////////////////////////////
	//			p2[0] = _verticesBump[(i + 1)*BumpWidth + j]._Tx;
	//			p2[1] = _verticesBump[(i + 1)*BumpWidth + j]._Ty;
	//			p2[2] = _verticesBump[(i + 1)*BumpWidth + j]._Tz;
	//
	//			p3[0] = _verticesBump[i*BumpWidth + (j + 1)]._Tx;
	//			p3[1] = _verticesBump[i*BumpWidth + (j + 1)]._Ty;
	//			p3[2] = _verticesBump[i*BumpWidth + (j + 1)]._Tz;
	//
	//			if (p2[2] != 0.0 && p3[2] != 0.0) {
	//				n_p1[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
	//				n_p1[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
	//				n_p1[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);
	//
	//				norm_n = (n_p1[0] * n_p1[0] + n_p1[1] * n_p1[1] + n_p1[2] * n_p1[2]);
	//
	//				if (norm_n != 0.0) {
	//					n_p1[0] = n_p1[0] / sqrt(norm_n);
	//					n_p1[1] = n_p1[1] / sqrt(norm_n);
	//					n_p1[2] = n_p1[2] / sqrt(norm_n);
	//
	//					n_tot++;
	//				}
	//			}
	//
	//			////////////////////////// Triangle 2 /////////////////////////////////
	//
	//			p2[0] = _verticesBump[i*BumpWidth + (j + 1)]._Tx;
	//			p2[1] = _verticesBump[i*BumpWidth + (j + 1)]._Ty;
	//			p2[2] = _verticesBump[i*BumpWidth + (j + 1)]._Tz;
	//
	//			p3[0] = _verticesBump[(i - 1)*BumpWidth + j]._Tx;
	//			p3[1] = _verticesBump[(i - 1)*BumpWidth + j]._Ty;
	//			p3[2] = _verticesBump[(i - 1)*BumpWidth + j]._Tz;
	//
	//			if (p2[2] != 0.0 && p3[2] != 0.0) {
	//				n_p2[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
	//				n_p2[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
	//				n_p2[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);
	//
	//				norm_n = (n_p2[0] * n_p2[0] + n_p2[1] * n_p2[1] + n_p2[2] * n_p2[2]);
	//
	//				if (norm_n != 0.0) {
	//					n_p2[0] = n_p2[0] / sqrt(norm_n);
	//					n_p2[1] = n_p2[1] / sqrt(norm_n);
	//					n_p2[2] = n_p2[2] / sqrt(norm_n);
	//
	//					n_tot++;
	//				}
	//			}
	//
	//			////////////////////////// Triangle 3 /////////////////////////////////
	//
	//			p2[0] = _verticesBump[(i - 1)*BumpWidth + j]._Tx;
	//			p2[1] = _verticesBump[(i - 1)*BumpWidth + j]._Ty;
	//			p2[2] = _verticesBump[(i - 1)*BumpWidth + j]._Tz;
	//
	//			p3[0] = _verticesBump[i*BumpWidth + (j - 1)]._Tx;
	//			p3[1] = _verticesBump[i*BumpWidth + (j - 1)]._Ty;
	//			p3[2] = _verticesBump[i*BumpWidth + (j - 1)]._Tz;
	//
	//			if (p2[2] != 0.0 && p3[2] != 0.0) {
	//				n_p3[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
	//				n_p3[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
	//				n_p3[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);
	//
	//				norm_n = (n_p3[0] * n_p3[0] + n_p3[1] * n_p3[1] + n_p3[2] * n_p3[2]);
	//
	//				if (norm_n != 0) {
	//					n_p3[0] = n_p3[0] / sqrt(norm_n);
	//					n_p3[1] = n_p3[1] / sqrt(norm_n);
	//					n_p3[2] = n_p3[2] / sqrt(norm_n);
	//
	//					n_tot++;
	//				}
	//			}
	//
	//			////////////////////////// Triangle 4 /////////////////////////////////
	//
	//			p2[0] = _verticesBump[i*BumpWidth + (j - 1)]._Tx;
	//			p2[1] = _verticesBump[i*BumpWidth + (j - 1)]._Ty;
	//			p2[2] = _verticesBump[i*BumpWidth + (j - 1)]._Tz;
	//
	//			p3[0] = _verticesBump[(i + 1)*BumpWidth + j]._Tx;
	//			p3[1] = _verticesBump[(i + 1)*BumpWidth + j]._Ty;
	//			p3[2] = _verticesBump[(i + 1)*BumpWidth + j]._Tz;
	//
	//			if (p2[2] != 0.0 && p3[2] != 0.0) {
	//				n_p4[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
	//				n_p4[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
	//				n_p4[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);
	//
	//				norm_n = (n_p4[0] * n_p4[0] + n_p4[1] * n_p4[1] + n_p4[2] * n_p4[2]);
	//
	//				if (norm_n != 0) {
	//					n_p4[0] = n_p4[0] / sqrt(norm_n);
	//					n_p4[1] = n_p4[1] / sqrt(norm_n);
	//					n_p4[2] = n_p4[2] / sqrt(norm_n);
	//
	//					n_tot++;
	//				}
	//			}
	//
	//			if (n_tot == 0) {
	//				currVBP->_TNx = 0.0;
	//				currVBP->_TNy = 0.0;
	//				currVBP->_TNz = 0.0;
	//				continue;
	//			}
	//
	//			n_p[0] = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0]) / float(n_tot);
	//			n_p[1] = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1]) / float(n_tot);
	//			n_p[2] = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2]) / float(n_tot);
	//
	//			norm_n = sqrt(n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);
	//
	//			if (norm_n != 0) {
	//				currVBP->_TNx = n_p[0] / norm_n;
	//				currVBP->_TNy = n_p[1] / norm_n;
	//				currVBP->_TNz = n_p[2] / norm_n;
	//			}
	//			else {
	//				currVBP->_TNx = 0.0;
	//				currVBP->_TNy = 0.0;
	//				currVBP->_TNz = 0.0;
	//			}
	//
	//		}
	//	}
}

void HeadOffV2::InitLabels() {
	cl_int ret;
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _BumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _Bump.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _LabelsMaskCL, true, origin, region, BumpWidth * 4 * sizeof(unsigned char), 0, _LabelsMask.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _WeightMapCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _WeightMap.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");


	/**********************************Initialise Bump attributes*******************************************/
	float colfill[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	ret = clEnqueueFillImage(_queue[BUMP_KER], _VMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage VMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _NMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage NMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _RGBMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage RGBMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _RGBMapBumpSwapCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage RGBMapBumpCL");
}

void HeadOffV2::ComputeLabels(MyMesh *TheMesh) {

	//for (int i = 0; i < 3818; i++) {
	//	TheMesh->_vertices[FrontIndices[i]]->_BackPoint = true;
		//cout << "i: " << i << " " << FrontIndices[i] << endl;
	//}

	cv::Mat img0(BumpHeight, BumpWidth, CV_16UC3);
	cv::Mat img1(BumpHeight, BumpWidth, CV_16UC3);
	float weights[3];
	Face *CurrFace;
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			float lab_val = -1.0f;
			unsigned short iii = 0;
			for (vector<Face *>::iterator it = TheMesh->_triangles.begin(); it != TheMesh->_triangles.end(); it++) {
				if (IsInTriangle(TheMesh, (*it), i, j)) {
					lab_val = float(iii);
					break;
				}
				iii++;
			}

			if (lab_val > -1.0f) {
				CurrFace = TheMesh->_triangles[int(lab_val)];
				TextUV s1 = TextUV(TheMesh->_uvs[CurrFace->_t1]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t1]->_v*float(BumpWidth));
				TextUV s2 = TextUV(TheMesh->_uvs[CurrFace->_t2]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t2]->_v*float(BumpWidth));
				TextUV s3 = TextUV(TheMesh->_uvs[CurrFace->_t3]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t3]->_v*float(BumpWidth));
				getWeightsB(weights, float(i), float(j), s1, s2, s3);

				img0.at<cv::Vec3w>(i, j)[0] = unsigned short(65535.0f * weights[0] / (weights[0] + weights[1] + weights[2]));
				img0.at<cv::Vec3w>(i, j)[1] = unsigned short(65535.0f * weights[1] / (weights[0] + weights[1] + weights[2]));
				img0.at<cv::Vec3w>(i, j)[2] = unsigned short(65535.0f * weights[2] / (weights[0] + weights[1] + weights[2]));
				img1.at<cv::Vec3w>(i, j)[0] = unsigned short(lab_val + 1.0f);
				img1.at<cv::Vec3w>(i, j)[1] = 0;
				img1.at<cv::Vec3w>(i, j)[2] = 0;
				//bool ok = TheMesh->_vertices[CurrFace->_v1]->_BackPoint && TheMesh->_vertices[CurrFace->_v2]->_BackPoint && TheMesh->_vertices[CurrFace->_v3]->_BackPoint;
				//if (ok)
				//	img1.at<cv::Vec3w>(i, j)[0] = 65535;
			}
			else {
				img0.at<cv::Vec3w>(i, j)[0] = 0;
				img0.at<cv::Vec3w>(i, j)[1] = 0;
				img0.at<cv::Vec3w>(i, j)[2] = 0;
				img1.at<cv::Vec3w>(i, j)[0] = 0;
				img1.at<cv::Vec3w>(i, j)[1] = 0;
				img1.at<cv::Vec3w>(i, j)[2] = 0;
			}
		}
	}
	//cv::imshow("img0", img0);
	cv::imwrite("Weights-240.png", img0);
	cv::imwrite("Labels-240.png", img1);
	//cv::imwrite("Front.png", img1);
	cv::waitKey(1);

}

void HeadOffV2::Register(vector<MyMesh *> Blendshape) {

	///////////////////////////////////////////
	_BlendshapeCoeff[0] = 1.0;
	for (int i = 1; i < 49; i++) {
		_BlendshapeCoeff[i] = 0.0;
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * j + i] = 0.0f;
		}
		_Pose[12 + i] = 0.0f;
	}
	_Pose[0] = 1.0f; _Pose[5] = 1.0f; _Pose[10] = 1.0f; _Pose[15] = 1.0f;

	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };

	// Compute Vertex map
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	cl_int ret = clEnqueueNDRangeKernel(_queue[VMAPBUMP_KER], _kernels[VMAPBUMP_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[VMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");
	ret = clEnqueueReadImage(_queue[VMAPBUMP_KER], _VMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _VMapBump.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read VMAPBUMP_KER output");

	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[NMAPBUMP_KER], _NMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _NMapBump.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read NMAPBUMP_KER output");

	///////////////////////////////////////////

	float pt[3];

	int iter = 0;
	bool converged = false;

	float pos[3];
	float nmle[3];
	float nmleD[3];
	//Point3DGPU *CurrV;
	float DepthP[3];
	float DepthN[3];

	int p_indx[2];
	float Mat[27];
	bool found_coresp = false;
	float min_dist = 1000.0;
	float pointClose[3];
	float pointCoord[3];
	int li, ui, lj, uj;
	float dist;
	float dist_angle;

	float weight;
	float JD[18];
	float JRot[18];
	float row[7];

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
	Eigen::Matrix<double, 6, 1> b;
	int shift = 0;
	double det;

	Eigen::Matrix<double, 6, 1> result;
	double q[4];
	double norm;

	double tmp[3][3];

	Eigen::Matrix3f Rinc;
	Eigen::Vector3f tinc;
	int step;

	float weight_landmarks = 10.0;

	while (iter < 10 /*&& !converged*/) {
		/*** Compute correspondences ***/
		for (int k = 0; k < 27; k++)
			Mat[k] = 0.0;

		step = iter < 2 ? 3 - iter : 2;
		for (int u = 0; u < BumpHeight - 1; u += step) {
			for (int v = 0; v < BumpWidth - 1; v += step) {

				if ((_NMapBump.at<cv::Vec4f>(u, v)[0] == 0.0 && _NMapBump.at<cv::Vec4f>(u, v)[1] == 0.0 && _NMapBump.at<cv::Vec4f>(u, v)[2] == 0.0))
				{
					continue;
				}

				pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
				pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
				pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

				nmle[0] = _NMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2);
				nmle[1] = _NMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2);
				nmle[2] = _NMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2);

				if (nmle[2] < 0.0)
					continue;

				// Search for corresponding point
				found_coresp = false;
				min_dist = 1000.0;

				/*** Projective association ***/

				// Project the point onto the depth image
				p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_intrinsic[0] + _intrinsic[2]))));
				p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_intrinsic[1] + _intrinsic[3]))));

				li = max((cDepthHeight - p_indx[1] - 1) - 1, 0);
				ui = min((cDepthHeight - p_indx[1] - 1) + 2, cDepthHeight);
				lj = max(p_indx[0] - 1, 0);
				uj = min(p_indx[0] + 2, cDepthWidth);

				for (int i = li; i < ui; i++) {
					for (int j = lj; j < uj; j++) {
						DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];
						DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
						DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
						DepthN[0] = _NMap.at<cv::Vec4f>(i, j)[0];
						DepthN[1] = _NMap.at<cv::Vec4f>(i, j)[1];
						DepthN[2] = _NMap.at<cv::Vec4f>(i, j)[2];
						if (DepthN[0] == 0.0 && DepthN[1] == 0.0 && DepthN[2] == 0.0f)
							continue;

						dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));
						dist_angle = nmle[0] * DepthN[0] + nmle[1] * DepthN[1] + nmle[2] * DepthN[2];

						if (dist < min_dist && dist_angle > 0.8) {
							min_dist = dist;
							pointClose[0] = DepthP[0];
							pointClose[1] = DepthP[1];
							pointClose[2] = DepthP[2];
						}
					}
				}

				if (min_dist < 0.01)
					found_coresp = true;

				if (found_coresp)
				{
					weight = 1.0; //0.0012/(0.0012 + 0.0019*(s[3]-0.4)*(s[3]-0.4));

					JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0;		JD[12] = 2.0*pt[2];	JD[15] = -2.0*pt[1];
					JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -2.0*pt[2]; JD[13] = 0.0;		JD[16] = 2.0*pt[0];
					JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = 2.0*pt[1];	JD[14] = -2.0*pt[0]; JD[17] = 0.0;

					JRot[0] = 0.0; JRot[3] = 0.0; JRot[6] = 0.0;	JRot[9] = 0.0;			JRot[12] = 2.0*nmle[2];	JRot[15] = -2.0*nmle[1];
					JRot[1] = 0.0; JRot[4] = 0.0; JRot[7] = 0.0;	JRot[10] = -2.0*nmle[2];	JRot[13] = 0.0;			JRot[16] = 2.0*nmle[0];
					JRot[2] = 0.0; JRot[5] = 0.0; JRot[8] = 0.0;	JRot[11] = 2.0*nmle[1];	JRot[14] = -2.0*nmle[0];	JRot[17] = 0.0;

					row[0] = weight*(-(nmle[0] * JD[0] + nmle[1] * JD[1] + nmle[2] * JD[2]) + JRot[0] * (pointClose[0] - pt[0]) + JRot[1] * (pointClose[1] - pt[1]) + JRot[2] * (pointClose[2] - pt[2]));
					row[1] = weight*(-(nmle[0] * JD[3] + nmle[1] * JD[4] + nmle[2] * JD[5]) + JRot[3] * (pointClose[0] - pt[0]) + JRot[4] * (pointClose[1] - pt[1]) + JRot[5] * (pointClose[2] - pt[2]));
					row[2] = weight*(-(nmle[0] * JD[6] + nmle[1] * JD[7] + nmle[2] * JD[8]) + JRot[6] * (pointClose[0] - pt[0]) + JRot[7] * (pointClose[1] - pt[1]) + JRot[8] * (pointClose[2] - pt[2]));
					row[3] = weight*(-(nmle[0] * JD[9] + nmle[1] * JD[10] + nmle[2] * JD[11]) + JRot[9] * (pointClose[0] - pt[0]) + JRot[10] * (pointClose[1] - pt[1]) + JRot[11] * (pointClose[2] - pt[2]));
					row[4] = weight*(-(nmle[0] * JD[12] + nmle[1] * JD[13] + nmle[2] * JD[14]) + JRot[12] * (pointClose[0] - pt[0]) + JRot[13] * (pointClose[1] - pt[1]) + JRot[14] * (pointClose[2] - pt[2]));
					row[5] = weight*(-(nmle[0] * JD[15] + nmle[1] * JD[16] + nmle[2] * JD[17]) + JRot[15] * (pointClose[0] - pt[0]) + JRot[16] * (pointClose[1] - pt[1]) + JRot[17] * (pointClose[2] - pt[2]));

					row[6] = -weight*(nmle[0] * (pointClose[0] - pt[0]) + nmle[1] * (pointClose[1] - pt[1]) + nmle[2] * (pointClose[2] - pt[2]));

					shift = 0;
					for (int k = 0; k < 6; ++k)        //rows
					{
						for (int l = k; l < 7; ++l)          // cols + b
						{
							Mat[shift] = Mat[shift] + row[k] * row[l];
							shift++;
						}
					}
				}
			}
		}

		// Add correspondences from facial features
		for (int i = 0; i < 43; i++) {
			MyPoint *DepthV = Blendshape[0]->Landmark(i);

			int u_curr = round(DepthV->_u*float(BumpHeight));
			int v_curr = round(DepthV->_v*float(BumpWidth));

			if ((_NMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] == 0.0 && _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] == 0.0 && _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] == 0.0) ||
				_landmarks.rows < i - 1)
			{
				continue;
			}

			pt[0] = _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

			nmle[0] = _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(0, 2);
			nmle[1] = _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(1, 2);
			nmle[2] = _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[1] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(u_curr, v_curr)[2] * _Rotation_inv(2, 2);

			if (nmle[2] < 0.0)
				continue;

			int indx_i = Myround(_landmarks.at<float>(1, i));
			int indx_j = Myround(_landmarks.at<float>(0, i));

			DepthP[0] = _VMap.at<cv::Vec4f>(indx_i, indx_j)[0];  // _vertices[i*cDepthWidth + j];
			DepthP[1] = _VMap.at<cv::Vec4f>(indx_i, indx_j)[1];
			DepthP[2] = _VMap.at<cv::Vec4f>(indx_i, indx_j)[2];
			DepthN[0] = _NMap.at<cv::Vec4f>(indx_i, indx_j)[0];
			DepthN[1] = _NMap.at<cv::Vec4f>(indx_i, indx_j)[1];
			DepthN[2] = _NMap.at<cv::Vec4f>(indx_i, indx_j)[2];
			if (DepthN[0] == 0.0 && DepthN[1] == 0.0 && DepthN[2] == 0.0f)
				continue;

			dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));
			if (dist > 0.01)
				continue;

			JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0;		JD[12] = 2.0*pt[2];	JD[15] = -2.0*pt[1];
			JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -2.0*pt[2]; JD[13] = 0.0;		JD[16] = 2.0*pt[0];
			JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = 2.0*pt[1];	JD[14] = -2.0*pt[0]; JD[17] = 0.0;

			JRot[0] = 0.0; JRot[3] = 0.0; JRot[6] = 0.0;	JRot[9] = 0.0;			JRot[12] = 2.0*nmle[2];	JRot[15] = -2.0*nmle[1];
			JRot[1] = 0.0; JRot[4] = 0.0; JRot[7] = 0.0;	JRot[10] = -2.0*nmle[2];	JRot[13] = 0.0;			JRot[16] = 2.0*nmle[0];
			JRot[2] = 0.0; JRot[5] = 0.0; JRot[8] = 0.0;	JRot[11] = 2.0*nmle[1];	JRot[14] = -2.0*nmle[0];	JRot[17] = 0.0;

			row[0] = weight*(-(nmle[0] * JD[0] + nmle[1] * JD[1] + nmle[2] * JD[2]) + JRot[0] * (DepthV->_x - pt[0]) + JRot[1] * (DepthV->_y - pt[1]) + JRot[2] * (DepthV->_z - pt[2]));
			row[1] = weight*(-(nmle[0] * JD[3] + nmle[1] * JD[4] + nmle[2] * JD[5]) + JRot[3] * (DepthV->_x - pt[0]) + JRot[4] * (DepthV->_y - pt[1]) + JRot[5] * (DepthV->_z - pt[2]));
			row[2] = weight*(-(nmle[0] * JD[6] + nmle[1] * JD[7] + nmle[2] * JD[8]) + JRot[6] * (DepthV->_x - pt[0]) + JRot[7] * (DepthV->_y - pt[1]) + JRot[8] * (DepthV->_z - pt[2]));
			row[3] = weight*(-(nmle[0] * JD[9] + nmle[1] * JD[10] + nmle[2] * JD[11]) + JRot[9] * (DepthV->_x - pt[0]) + JRot[10] * (DepthV->_y - pt[1]) + JRot[11] * (DepthV->_z - pt[2]));
			row[4] = weight*(-(nmle[0] * JD[12] + nmle[1] * JD[13] + nmle[2] * JD[14]) + JRot[12] * (DepthV->_x - pt[0]) + JRot[13] * (DepthV->_y - pt[1]) + JRot[14] * (DepthV->_z - pt[2]));
			row[5] = weight*(-(nmle[0] * JD[15] + nmle[1] * JD[16] + nmle[2] * JD[17]) + JRot[15] * (DepthV->_x - pt[0]) + JRot[16] * (DepthV->_y - pt[1]) + JRot[17] * (DepthV->_z - pt[2]));

			row[6] = -weight_landmarks*(nmle[0] * (DepthP[0] - pt[0]) + nmle[1] * (DepthP[1] - pt[1]) + nmle[2] * (DepthP[2] - pt[2]));

			shift = 0;
			for (int k = 0; k < 6; ++k)        //rows
			{
				for (int l = k; l < 7; ++l)          // cols + b
				{
					Mat[shift] = Mat[shift] + row[k] * row[l];
					shift++;
				}
			}
		}

		shift = 0;
		for (int i = 0; i < 6; ++i) {  //rows
			for (int j = i; j < 7; ++j)    // cols + b
			{
				double value = double(Mat[shift++]);
				if (j == 6)       // vector b
					b(i) = value;
				else
					A(j, i) = A(i, j) = value;
			}
		}

		//checking nullspace
		det = A.determinant();

		if (fabs(det) < 1e-15 || det != det)
		{
			if (det != det) std::cout << "qnan" << endl;
			std::cout << "det null" << endl;
			return;
		}

		result = A.llt().solve(b);
		norm = (result(3)*result(3) + result(4)*result(4) + result(5)*result(5));
		q[1] = result(3); q[2] = result(4); q[3] = result(5); q[0] = sqrt(1 - norm);

		quaternion2matrix(q, tmp);

		Rinc(0, 0) = float(tmp[0][0]); Rinc(0, 1) = float(tmp[0][1]); Rinc(0, 2) = float(tmp[0][2]);
		Rinc(1, 0) = float(tmp[1][0]); Rinc(1, 1) = float(tmp[1][1]); Rinc(1, 2) = float(tmp[1][2]);
		Rinc(2, 0) = float(tmp[2][0]); Rinc(2, 1) = float(tmp[2][1]); Rinc(2, 2) = float(tmp[2][2]);
		tinc = result.head<3>().cast<float>();

		_Translation_inv = Rinc * _Translation_inv + tinc;
		_Rotation_inv = Rinc * _Rotation_inv;

		if (((Rinc - Eigen::Matrix3f::Identity()).norm() < 1.0e-6 && tinc.norm() < 1.0e-6)) {
			converged = true;
		}

		iter++;
	}

	if (VERBOSE)
		std::cout << "Rot: " << _Rotation_inv << " Translation: " << _Translation_inv;
	_Rotation = _Rotation_inv.inverse();
	_Translation = -_Rotation * _Translation_inv;

	cv::Mat Rotcv(3, 3, CV_32FC1);
	cv::Point3f Transcv;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * i + j] = _Rotation_inv(j, i);
			Rotcv.at<float>(i, j) = _Rotation_inv(i,j);
		}
		_Pose[12 + i] = _Translation_inv(i);
	}
	Transcv.x = _Translation_inv(0);
	Transcv.y = _Translation_inv(1);
	Transcv.z = _Translation_inv(2);



	for (vector<MyMesh *>::iterator it = Blendshape.begin(); it != Blendshape.end(); it++) {
		(*it)->Rotate(Rotcv);
		(*it)->Translate(Transcv);
		(*it)->AffectToTVal();
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * j + i] = 0.0f;
		}
		_Pose[12 + i] = 0.0f;
	}
	_Pose[0] = 1.0f; _Pose[5] = 1.0f; _Pose[10] = 1.0f; _Pose[15] = 1.0f;
	_Rotation_inv = Eigen::Matrix3f::Identity();
	_Translation_inv = Eigen::Vector3f::Zero();
	_Rotation = Eigen::Matrix3f::Identity();
	_Translation = Eigen::Vector3f::Zero();


	// Compute Vertex map

	/*ret = clEnqueueNDRangeKernel(_queue[ANIMATE_KER], _kernels[ANIMATE_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[ANIMATE_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");*/
}

void HeadOffV2::RegisterGPU(MyMesh *RefMesh) {

	// Compute Vertex map
	cl_int ret;
	int gws_x = divUp(1 + BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	size_t Rgws1[2] = { 28 * STRIDE, divUp(3 * 43 + BumpHeight*BumpWidth, (STRIDE * 2)) };
	size_t Rlws1[2] = { STRIDE, 1 };

	size_t Rgws2[2] = { 28 * STRIDE, divUp(Rgws1[1], (STRIDE * 2)) };
	size_t Rlws2[2] = { STRIDE, 1 };

	cl_event evts[3];

	int iter = 0;
	bool converged = false;

	float sum, prev_sum, count;
	float lambda = 1.0;


	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
	Eigen::Matrix<double, 6, 1> b;
	int shift = 0;
	double det;

	Eigen::Matrix<double, 6, 1> result;
	double q[4];
	double norm;

	double tmp[3][3];

	Eigen::Matrix3f Rinc;
	Eigen::Vector3f tinc;

	int length_out, length_out2;
	int length = gws_x*gws_y;
	int length2;

	for (int lvl = _lvl - 1; lvl > -1; lvl--) {

		int fact = int(pow((float)2.0, (int)lvl));
		ret = clSetKernelArg(_kernels[GICP_KER], 6, sizeof(int), &fact);

		int ttt = divUp(BumpHeight, fact);
		gws_x = divUp(ttt, THREAD_SIZE_X);
		ttt = divUp(BumpWidth, fact);
		gws_y = divUp(ttt, THREAD_SIZE_Y);
		gws[0] = 1+gws_x*THREAD_SIZE_X;
		gws[1] = gws_y*THREAD_SIZE_Y;

		iter = 1;
		converged = iter > _max_iter[lvl];

		Eigen::Vector3f Translation_prev;
		Eigen::Matrix3f Rotation_prev;
		prev_sum = 0.0f;

		while (!converged) {
			/*** Compute correspondences ***/

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					_Pose[4 * j + i] = float(_Rotation_inv(i, j));
				}
				_Pose[12 + i] = float(_Translation_inv(i));
			}

			_nbMatches = 0;

			ret = clEnqueueNDRangeKernel(_queue[GICP_KER], _kernels[GICP_KER], 2, NULL, gws, lws, 0, NULL, &evts[0]);
			checkErr(ret, "GICP_KER::enqueueNDRangeKernel()");

			ret = clEnqueueReadBuffer(_queue[GICP_KER], _NbMatchesCL, true, 0, sizeof(int), &_nbMatches, 1, &evts[0], NULL);
			checkErr(ret, "Unable to read output");

			Rgws1[1] = divUp(_nbMatches, (STRIDE * 2));
			Rgws2[1] = divUp(Rgws1[1], (STRIDE * 2));

			length = _nbMatches;
			length_out = Rgws1[1];
			ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 5, sizeof(int), &length);
			ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 6, sizeof(int), &length_out);

			ret = clEnqueueNDRangeKernel(_queue[REDUCEGICP_KER], _kernels[REDUCEGICP_KER], 2, NULL, Rgws1, Rlws1, 0, NULL, &evts[1]);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() REDUCE1_KER");

			length2 = length_out;
			length_out2 = divUp(length2, (STRIDE * 2));
			ret = clSetKernelArg(_kernels[REDUCE2_KER], 2, sizeof(int), &length2);
			ret = clSetKernelArg(_kernels[REDUCE2_KER], 3, sizeof(int), &length_out2);

			ret = clEnqueueNDRangeKernel(_queue[REDUCE2_KER], _kernels[REDUCE2_KER], 2, NULL, Rgws2, Rlws2, 1, &evts[1], &evts[2]);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() REDUCE2_KER");

			ret = clEnqueueReadBuffer(_queue[REDUCE2_KER], _bufReduce2CL, true, 0, 28 * sizeof(float), _outbuffReduce, 1, &evts[2], NULL);
			checkErr(ret, "Unable to read output");

			sum = _outbuffReduce[27];
			sum = sum / float(_nbMatches);

			shift = 0;
			for (int i = 0; i < 6; ++i) {  //rows
				for (int j = i; j < 7; ++j)    // cols + b
				{
					double value = double(_outbuffReduce[shift++]);
					if (j == 6)       // vector b
						b(i) = value;
					else
						A(j, i) = A(i, j) = value;
				}
				I(i, i) = A(i, i);
			}

			det = A.determinant();

			if (fabs(det) < 1e-15 || det != det)
			{
				if (det != det) std::cout << "qnan" << endl;
				std::cout << "det null" << endl;
				return;
			}


			//for (int k = 0; k < 6; k++)
			//	I(k, k) = A(k, k) * 100.0 / double(1.0f + 100000.0f*fabs(sum - prev_sum));

			if (prev_sum != 0.0f) {
				if (sum > prev_sum) {
					_Translation_inv = Translation_prev;
					_Rotation_inv = Rotation_prev;
					lambda = lambda / 1.5;
					iter++;
					converged = iter > _max_iter[lvl];
					continue;
				}
				else {
					if (sum < prev_sum) {
						Translation_prev = _Translation_inv;
						Rotation_prev = _Rotation_inv;
						prev_sum = sum;
					}
				}
			}
			else {
				//prev_sum = sum;
				Translation_prev = _Translation_inv;
				Rotation_prev = _Rotation_inv;
			}

			//result = (A + I).llt().solve(b);
			//cout << result << endl;
			result = A.llt().solve(b);
			norm = (result(3)*result(3) + result(4)*result(4) + result(5)*result(5));
			if (norm > 1.0) {
				result(0) = result(1) = result(2) = result(3) = result(4) = result(5) = 0.0;
			}
			q[1] = result(3); q[2] = result(4); q[3] = result(5); q[0] = sqrt(1.0 - norm);

			quaternion2matrix(q, tmp);

			Rinc(0, 0) = float(tmp[0][0]); Rinc(0, 1) = float(tmp[0][1]); Rinc(0, 2) = float(tmp[0][2]);
			Rinc(1, 0) = float(tmp[1][0]); Rinc(1, 1) = float(tmp[1][1]); Rinc(1, 2) = float(tmp[1][2]);
			Rinc(2, 0) = float(tmp[2][0]); Rinc(2, 1) = float(tmp[2][1]); Rinc(2, 2) = float(tmp[2][2]);
			tinc = result.head<3>().cast<float>();

			_Translation_inv = Rinc * _Translation_inv + tinc;
			_Rotation_inv = Rinc * _Rotation_inv;
			iter++;

			if (iter > _max_iter[lvl] || ((Rinc - Eigen::Matrix3f::Identity()).norm() < 1.0e-6 && tinc.norm() < 1.0e-6)) {
				converged = true;
			}

		}
	}

	if (VERBOSE)
		std::cout << "Rot: " << _Rotation_inv << " Translation: " << _Translation_inv;
	_Rotation = _Rotation_inv.inverse();
	_Translation = -_Rotation * _Translation_inv;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * i + j] = _Rotation_inv(j, i);
		}
		_Pose[12 + i] = _Translation_inv(i);
	}

	_TranslationWindow.push_back(_Translation);
	_RotationWindow.push_back(_Rotation);

	if (_TranslationWindow.size() > 10) {
		_TranslationWindow.erase(_TranslationWindow.begin());
		_RotationWindow.erase(_RotationWindow.begin());
	}


	if (save_data) {
		ofstream  filestr;

		string filename = string(dest_name) + string("\\Animation\\Pose") + to_string(_idx) + string(".txt");
		filestr.open(filename, fstream::out);
		while (!filestr.is_open()) {
			cout << "Could not open MappingList" << endl;
			return;
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				filestr << _Pose[4 * i + j] << endl;
			}
		}

		filestr.close();
	}
}

void HeadOffV2::EstimateBlendShapeCoefficientsPR(vector<MyMesh *> Blendshape) {

	///// Load data from gpu ////
	float *Vtxtemp = (float *)malloc(49 * 6 * BumpHeight * BumpWidth*sizeof(float));
	cl_int ret = clEnqueueReadBuffer(_queue[DATAPROC_KER], _verticesBSCL, true, 0, 49 * 6 * BumpHeight * BumpWidth * sizeof(float), Vtxtemp, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");


	/* 1. Compute matrix B **************************/
	// Component from the geometry
	vector<double *> B;
	vector<double> c;
	Eigen::VectorXd xres = Eigen::VectorXd(27);
	for (int i = 1; i < 28; i++) {
		xres(i - 1) = double(_BlendshapeCoeff[i]);
	}

	float pt[3];
	float pt_T[3];
	float nmle[3];
	float pt_corr[3];
	float nmle_corr[3];
	int p_indx[2];
	float dist;
	float d;
	int tid;

	double w = 5.0e-5;

	bool first_match = true;
	bool converged = false;
	int max_iter = 10;
	int iter = 0;
	while (!converged) {
		for (int u = 0; u < BumpHeight - 1; u++) {
			for (int v = 0; v < BumpWidth - 1; v++) {
				tid = u*BumpWidth + v;
				if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f)
					continue;

				pt[0] = Vtxtemp[6 * tid];
				pt[1] = Vtxtemp[6 * tid + 1];
				pt[2] = Vtxtemp[6 * tid + 2];
				nmle[0] = Vtxtemp[6 * tid + 3];
				nmle[1] = Vtxtemp[6 * tid + 4];
				nmle[2] = Vtxtemp[6 * tid + 5];

				for (int k = 1; k < 28; k++) {
					// This blended normal is not really a normal since it may be not normalized
					nmle[0] = nmle[0] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3] * float(xres(k - 1));
					nmle[1] = nmle[1] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4] * float(xres(k - 1));
					nmle[2] = nmle[2] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5] * float(xres(k - 1));

					pt[0] = pt[0] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid] * float(xres(k - 1));
					pt[1] = pt[1] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] * float(xres(k - 1));
					pt[2] = pt[2] + Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] * float(xres(k - 1));
				}
				d = _Bump.at<cv::Vec4f>(u, v)[0] / 1000.0f;
				/*cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] << endl;
				cout << "nmle " << nmle[0] << " " << nmle[1] << " " << nmle[2] << endl;
				cout << "d " << d << endl;*/

				pt_T[0] = pt[0] + d*nmle[0];
				pt_T[1] = pt[1] + d*nmle[1];
				pt_T[2] = pt[2] + d*nmle[2];

				pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
				pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
				pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

				_VMapBump.at<cv::Vec4f>(u, v)[0] = pt[0];
				_VMapBump.at<cv::Vec4f>(u, v)[1] = pt[1];
				_VMapBump.at<cv::Vec4f>(u, v)[2] = pt[2];

				p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_intrinsic[0] + _intrinsic[2]))));
				p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_intrinsic[1] + _intrinsic[3]))));

				int li = max((cDepthHeight - p_indx[1] - 1), 0); //max((cDepthHeight - p_indx[1] - 1) - 5, 0);
				int ui = min((cDepthHeight - p_indx[1] - 1) + 1, cDepthHeight);  //min((cDepthHeight - p_indx[1] - 1) + 6, cDepthHeight);
				int lj = max(p_indx[0], 0); //max(p_indx[0] - 5, 0);
				int uj = min(p_indx[0] + 1, cDepthWidth); //min(p_indx[0] + 6, cDepthWidth);
				int best_i, best_j;

				float min_dist = 1.0e6;
				float DepthP[3];
				float DepthN[3];
				for (int i = li; i < ui; i++) {
					for (int j = lj; j < uj; j++) {
						DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];  // _vertices[i*cDepthWidth + j];
						DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
						DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
						DepthN[0] = _NMap.at<cv::Vec4f>(i, j)[0];
						DepthN[1] = _NMap.at<cv::Vec4f>(i, j)[1];
						DepthN[2] = _NMap.at<cv::Vec4f>(i, j)[2];
						if (DepthN[0] == 0.0 && DepthN[1] == 0.0 && DepthN[2] == 0.0f)
							continue;

						dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));
						//dist_angle = nmle[0] * DepthN[0] + nmle[1] * DepthN[1] + nmle[2] * DepthN[2];

						if (dist < min_dist/* && dist_angle > 0.8*/) {
							min_dist = dist;
							pt_corr[0] = DepthP[0];
							pt_corr[1] = DepthP[1];
							pt_corr[2] = DepthP[2];
							nmle_corr[0] = DepthN[0];
							nmle_corr[1] = DepthN[1];
							nmle_corr[2] = DepthN[2];
							best_i = i;
							best_j = j;
						}
					}
				}

				/*pt_corr[0] = _VMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[0];
				pt_corr[1] = _VMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[1];
				pt_corr[2] = _VMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[2];

				nmle_corr[0] = _NMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[0];
				nmle_corr[1] = _NMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[1];
				nmle_corr[2] = _NMap.at<cv::Vec4f>(cDepthHeight - p_indx[1] - 1, p_indx[0])[2];

				if (nmle_corr[0] == 0.0f && nmle_corr[1] == 0.0f && nmle_corr[2] == 0.0f)
					continue;

				dist = sqrt((pt_corr[0] - pt[0])*(pt_corr[0] - pt[0]) + (pt_corr[1] - pt[1])*(pt_corr[1] - pt[1]) + (pt_corr[2] - pt[2])*(pt_corr[2] - pt[2]));*/

				if (min_dist > 0.005)
					continue;
				
				/*if (tid > 105400 && tid%100 == 0) {
					_RGBMapBump.at<cv::Vec4f>(u, v)[0] = 255.0f;
					_RGBMapBump.at<cv::Vec4f>(u, v)[1] = 0.0f;
					_RGBMapBump.at<cv::Vec4f>(u, v)[2] = 0.0f;
					_RGBMap.at<cv::Vec4b>(best_i, best_j)[2] = 0;
					_RGBMap.at<cv::Vec4b>(best_i, best_j)[1] = 255;
					_RGBMap.at<cv::Vec4b>(best_i, best_j)[0] = 0;
					cout << "tid: " << tid << endl;
					clock_t last_time = clock();
					clock_t current_time = clock();
					while ((current_time - last_time) / CLOCKS_PER_SEC < 1.0) {
						current_time = clock();
					}
				}*/

				double *tmp = new double[27];
				//cout << "tmp ";
				for (int k = 1; k < 28; k++) {
					pt_T[0] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3];
					pt_T[1] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4];
					pt_T[2] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5];
					pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
					pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
					pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);
					//(nT((bj-bo) + d(nj-n0))
					tmp[k - 1] = double(nmle_corr[0] * pt[0] + nmle_corr[1] * pt[1] + nmle_corr[2] * pt[2]);
					//cout << tmp[k - 1] << " ";
				}
				//cout << " " << endl;
				B.push_back(tmp);
				pt_T[0] = Vtxtemp[6 * tid] + d * Vtxtemp[6 * tid + 3];
				pt_T[1] = Vtxtemp[6 * tid + 1] + d * Vtxtemp[6 * tid + 4];
				pt_T[2] = Vtxtemp[6 * tid + 2] + d * Vtxtemp[6 * tid + 5];
				pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
				pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
				pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2) + _Translation_inv(2);
				c.push_back(-double(nmle_corr[0] * (pt[0] - pt_corr[0]) +
									nmle_corr[1] * (pt[1] - pt_corr[1]) +
									nmle_corr[2] * (pt[2] - pt_corr[2])));
			}
		}

		//cout << "size B: " << B.size() << endl;

		// Append component from the landmarks
		for (int l = 0; l < 0; l++) {
			if (_landmarks.cols < l + 1)
				continue;

			int idx_i = _landmarksBump.at<int>(0, l);
			int idx_j = _landmarksBump.at<int>(1, l);

			if (idx_i < 0 || idx_i > BumpHeight - 1 || idx_j < 0 || idx_j > BumpWidth - 1)
				continue;

			if (_Bump.at<cv::Vec4f>(idx_i, idx_j)[1] == 0.0f)
				continue;

			tid = idx_i*BumpWidth + idx_j;
			d = _Bump.at<cv::Vec4f>(idx_i, idx_j)[0] / 1000.0f;

			float p_u = _landmarks.at<float>(1, l);
			float p_v = _landmarks.at<float>(0, l);

			double *tmpX = new double[48];
			double *tmpY = new double[48];
			//double *tmpZ = new double[48];
			for (int k = 1; k < 28; k++) {
				//(f((bj-bo) + d(nj-n0))
				pt_T[0] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3];
				pt_T[1] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4];
				pt_T[2] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5];
				pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
				pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
				pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);

				tmpX[k - 1] = w*double(pt[2] * (float(cDepthHeight) - 1.0 + _intrinsic[3] - p_u) - _intrinsic[1] * pt[1]);
				tmpY[k - 1] = w*double(pt[2] * (_intrinsic[0] - p_v) + _intrinsic[0] * pt[0]);
				//tmpZ[k - 1] = Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] + d * Vtxtemp[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5];
			}
			B.push_back(tmpX);
			B.push_back(tmpY);

			pt_T[0] = Vtxtemp[6 * tid] + d * Vtxtemp[6 * tid + 3];
			pt_T[1] = Vtxtemp[6 * tid + 1] + d * Vtxtemp[6 * tid + 4];
			pt_T[2] = Vtxtemp[6 * tid + 2] + d * Vtxtemp[6 * tid + 5];
			pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2) + _Translation_inv(2);
			c.push_back(w*double((p_u - (float(cDepthHeight) - 1.0 + _intrinsic[3]))*pt[2] + _intrinsic[1] * pt[1]));
			c.push_back(w*double((p_v - _intrinsic[2])*pt[2] - _intrinsic[0] * pt[0]));
		}

		/***2. Compute pseud inverse*************************************************/
		// Compute BTB
		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(27, 27); 
		Eigen::VectorXd cc = Eigen::VectorXd(27);

		for (int i = 0; i < 27; i++) {
			for (int j = i; j < 27; j++) {
				double val = 0.0;
				for (vector<double *>::iterator it = B.begin(); it != B.end(); it++) {
					val += (*it)[i] * (*it)[j];
				}
				Q(i, j) = Q(j, i) = val;
			}
			double val = 0.0;
			int k = 0;
			for (vector<double *>::iterator it = B.begin(); it != B.end(); it++) {
				val += (*it)[i] * c[k];
				k++;
			}
			cc(i) = val;
		}

		//cout << Q << endl; 
		//Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
		//cout << "Its singular values are:" << endl << svd.singularValues() << endl;
		//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
		//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
		//determinant
		double det = Q.determinant();
		//cout << "det" << det << endl;

		if (det == 0.0/*fabs(det) < 1e-15*/ || det != det)
		{
			if (det != det) std::cout << "qnan" << endl;
			std::cout << "det null" << endl;
			std::free(Vtxtemp);
			B.clear();
			c.clear();
			return;
		}

		/// Compute matrix PseudoInverse
		int n_lines = B.size();
		double *PseudInv = (double *)malloc(n_lines * 27 * sizeof(double));
		Eigen::MatrixXd Q_inv = Q.inverse();
		/*cout << Q_inv << endl;
		int tmpval;
		cin >> tmpval;*/

		//cout << Q_inv << endl;
		//Eigen::MatrixXd sigma_inv = Eigen::MatrixXd::Zero(27, 27);
		//for (int i = 0; i < 27; i++)
		//	sigma_inv(i, i) = 1.0 / svd.singularValues()(i);
		//Eigen::MatrixXd Q_invb = svd.matrixV() * sigma_inv * svd.matrixU().transpose();
		//cout << "with SVD: " << Q_invb << endl;


		for (int i = 0; i < 27; i++) {
			for (int j = 0; j < n_lines; j++) {
				double val = 0.0;
				for (int k = 0; k < 27; k++) {
					val += Q_inv(i, k) * B[j][k];
				}
				PseudInv[i*n_lines + j] = val;
			}
		}

		// Compute I - Pseudo * B
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(27, 27);
		for (int i = 0; i < 27; i++) {
			for (int j = 0; j < 27; j++) {
				double val = 0.0;
				int k = 0;
				for (vector<double *>::iterator it = B.begin(); it != B.end(); it++) {
					val += PseudInv[i*n_lines + k] * (*it)[j];
					k++;
				}
				A(i, j) = A(j, i) = -val;
			}
			A(i, i) = 1.0 + A(i, i);
		}

		Eigen::VectorXd x0 = Eigen::VectorXd(27);
		for (int i = 0; i < 27; i++) {
			double val = 0.0;
			int j = 0;
			for (vector<double>::iterator it = c.begin(); it != c.end(); it++) {
				val += PseudInv[i*n_lines + j] * (*it);
				j++;
			}

			double val1 = 0.0;
			for (int j = 0; j < 27; j++) {
				val1 += A(i, j);
			}

			x0(i) = val;// +val1;
		}
		/*cout << x0 << endl;
		cout << A << endl;
		int tmpval;
		cin >> tmpval;*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TEST Validity of initial solution ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//x0 = Q_inv * cc;
		//double Res = 0.0;
		//int k = 0;
		/*for (vector<double *>::iterator it = B.begin(); it != B.end(); it++) {
			double val = 0.0;
			for (int i = 0; i < 27; i++) {
				val += (*it)[i] * x0(i);
			}
			Res = max(Res, (val - c[k])*(val - c[k]));
			k++;
		}
		cout << "Residual: " << Res <<  " " << x0.transpose() << endl;
		int tmpval;
		cin >> tmpval;*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		//xres = ParallelRelaxation(Q_inv, x0, 0.0, 1.0);


		iter++;
		cout << "outer_iter: " << iter << " " << xres.transpose() << endl;
		converged = (iter > max_iter);
		for (int i = 0; i < 27; i++) {
			if (xres(i) < 0.0)
				xres(i) = 0.0;
			if (xres(i) > 1.0)
				xres(i) = 1.0;
		}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TEST Validity of final solution ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//Res = 0.0;
		//k = 0;
		//for (vector<double *>::iterator it = B.begin(); it != B.end(); it++) {
		//	double val = 0.0;
		//	for (int i = 0; i < 27; i++) {
		//		val += (*it)[i] * xres(i);
		//	}
		//	Res = max(Res, (val - c[k])*(val - c[k]));  //+= (val - c[k])*(val - c[k]);
		//	k++;
		//}
		//cout << "Residual: " << Res << endl;
		//cin >> tmpval;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		B.clear();
		c.clear();
	}
	std::free(Vtxtemp);

	cout << "_BlendshapeCoeff: " << endl;
	for (int i = 1; i < 28; i++) {
		_BlendshapeCoeff[i] = float(xres(i - 1));
		cout << _BlendshapeCoeff[i] << endl;
	}

	/*for (int i = 0; i < 43; i++) {
		int idx_i = Myround(Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._u*float(BumpHeight));
		int idx_j = Myround(Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._v*float(BumpWidth));

		float fact_BP = 1000.0f;
		float d = 0.0f;
		if (_Bump.at<cv::Vec4f>(idx_i, idx_j)[1] == 0.0f) {
			_Vtx[0][3 * i] = 0.0f;
			_Vtx[0][3 * i + 1] = 0.0f;
			_Vtx[0][3 * i + 2] = 0.0f;
			continue;
		}
		else {
			d = _Bump.at<cv::Vec4f>(idx_i, idx_j)[0] / fact_BP;
		}

		float tmpPt[3];
		tmpPt[0] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._x + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Nx;
		tmpPt[1] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._y + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Ny;
		tmpPt[2] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._z + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Nz;

		float ptRef[3];
		_Vtx[0][3 * i] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		_Vtx[0][3 * i + 1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		_Vtx[0][3 * i + 2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);
		ptRef[0] = _Vtx[0][3 * i]; ptRef[1] = _Vtx[0][3 * i + 1]; ptRef[2] = _Vtx[0][3 * i + 2];

		for (int k = 1; k < 28; k++) {
			tmpPt[0] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._x + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Nx;
			tmpPt[1] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._y + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Ny;
			tmpPt[2] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._z + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Nz;

			_Vtx[k][3 * i] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
			_Vtx[k][3 * i + 1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
			_Vtx[k][3 * i + 2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);
		}
	}*/
}

void HeadOffV2::EstimateBlendShapeCoefficientsPRGPU(vector<MyMesh *> Blendshape) {

	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(NB_BS - 1, NB_BS - 1);
	Eigen::VectorXd cc = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd x0 = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd xres = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd lb = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd ub = Eigen::VectorXd(NB_BS - 1);

	int fact, iter, max_iter = 0;
	bool converged = false;

	int size_tables = ((NB_BS)*(NB_BS + 1)) / 2;

	int gws_x = divUp(1 + BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	size_t Rgws1[2] = { size_tables * STRIDE, divUp(3 * 43 + BumpHeight * BumpWidth, (STRIDE * 2)) };
	size_t Rlws1[2] = { STRIDE, 1 };

	size_t Rgws2[2] = { size_tables * STRIDE, divUp(Rgws1[1], (STRIDE * 2)) };
	size_t Rlws2[2] = { STRIDE, 1 };

	int dim_x = 1;
	size_t gPws[2] = { divUp(dim_x, THREAD_SIZE_X)*THREAD_SIZE_X, divUp(dim_x, THREAD_SIZE_X)*THREAD_SIZE_Y };
	size_t lPws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	if (_idx_curr < 100/*!_landmarkOK/*_idx_curr < 10*/) {
		//SetLandmarks(Blendshape[0]);
		goto SAVE_BSCOEFF;
	}

	//if (_idx_curr == 1)
	//	SetLandmarks(Blendshape[0]);
	
	//for (int i = 0; i < 43; i++) {
	//	if (_landmarks.at<float>(0, i) == 0.0 && _landmarks.at<float>(1, i) == 0.0)
	//		goto SAVE_BSCOEFF;

	//	MyPoint *LandMark = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0],
	//		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1],
	//		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2]);
	//	if (LandMark->_x == 0.0f && LandMark->_y == 0.0f && LandMark->_z == 0.0f) 
	//		goto SAVE_BSCOEFF;
	//	
	//	//return;
	//}

	//int size_tables = ((NB_BS - 1)*(NB_BS - 2)) / 2 + 2 * (NB_BS - 1);

	cl_int ret;

	cl_event evts[3];

	clock_t current_time;

	/* 1. Compute matrix B **************************/
	// Component from the geometry

	converged = false;
	max_iter = 3;
	iter = 0;
	int shift;
	double value;
	fact = 1;
	ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 6, sizeof(int), &fact);
	ret = clSetKernelArg(_kernels[PSEUDOINV_KER], 1, sizeof(int), &fact);

	int length;
	int length_out;
	int length2;
	int length_out2;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * j + i] = float(_Rotation_inv(i, j));
		}
		_Pose[12 + i] = float(_Translation_inv(i));
	}

	//cout << "START" << endl;

	for (int lvl = _lvl - 1; lvl > -1; lvl--) {

		int fact = int(pow((float)2.0, (int)lvl));
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 6, sizeof(int), &fact);

		int ttt = divUp(BumpHeight, fact);
		gws_x = divUp(ttt, THREAD_SIZE_X);
		ttt = divUp(BumpWidth, fact);
		gws_y = divUp(ttt, THREAD_SIZE_Y);
		gws[0] = 1+gws_x*THREAD_SIZE_X;
		gws[1] = gws_y*THREAD_SIZE_Y;


		iter = 1;
		converged = iter > _max_iterPR[lvl];

		while (!converged) {
			_nbMatches = 0;
			//current_time = clock();
			ret = clEnqueueNDRangeKernel(_queue[SYSTEMPRB_KER], _kernels[SYSTEMPRB_KER], 2, NULL, gws, lws, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() SYSTEMPRB_KER");

			ret = clFinish(_queue[SYSTEMPRB_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			ret = clEnqueueReadBuffer(_queue[SYSTEMPRB_KER], _NbMatchesCL, true, 0, sizeof(int), &_nbMatches, 0, NULL, NULL);
			checkErr(ret, "Unable to read output");
			//cout << "_nbMatches: " << _nbMatches << endl;

			Rgws1[0] = size_tables * STRIDE;
			Rgws1[1] = divUp(_nbMatches, (STRIDE * 2));
			Rgws2[0] = size_tables * STRIDE;
			Rgws2[1] = divUp(Rgws1[1], (STRIDE * 2));

			length = _nbMatches;
			length_out = Rgws1[1];
			ret = clSetKernelArg(_kernels[REDUCE1_KER], 5, sizeof(int), &length);
			ret = clSetKernelArg(_kernels[REDUCE1_KER], 6, sizeof(int), &length_out);


			ret = clEnqueueNDRangeKernel(_queue[REDUCE1_KER], _kernels[REDUCE1_KER], 2, NULL, Rgws1, Rlws1, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() REDUCE1_KER");

			ret = clFinish(_queue[REDUCE1_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			length2 = length_out;
			length_out2 = divUp(length2, (STRIDE * 2));
			ret = clSetKernelArg(_kernels[REDUCE2_KER], 2, sizeof(int), &length2);
			ret = clSetKernelArg(_kernels[REDUCE2_KER], 3, sizeof(int), &length_out2);

			ret = clEnqueueNDRangeKernel(_queue[REDUCE2_KER], _kernels[REDUCE2_KER], 2, NULL, Rgws2, Rlws2, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() REDUCE2_KER");

			ret = clSetKernelArg(_kernels[PSEUDOINV_KER], 4, sizeof(int), &_nbMatches);

			dim_x = divUp(_nbMatches, 2);
			gPws[0] = divUp(dim_x, THREAD_SIZE_X)*THREAD_SIZE_X;
			gPws[1] = divUp(dim_x, THREAD_SIZE_Y)*THREAD_SIZE_Y;

			Rgws1[0] = (NB_BS - 1) * STRIDE;
			Rgws2[0] = (NB_BS - 1) * STRIDE;

			length = _nbMatches;
			length_out = divUp(length, (STRIDE * 2));
			ret = clSetKernelArg(_kernels[ATC_KER], 3, sizeof(int), &length);
			ret = clSetKernelArg(_kernels[ATC_KER], 4, sizeof(int), &length_out);

			length2 = length_out;
			length_out2 = divUp(length2, (STRIDE * 2));
			ret = clSetKernelArg(_kernels[REDSOLVE_KER], 2, sizeof(int), &length2);
			ret = clSetKernelArg(_kernels[REDSOLVE_KER], 3, sizeof(int), &length_out2);

			ret = clFinish(_queue[REDUCE2_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			ret = clEnqueueReadBuffer(_queue[REDUCE2_KER], _bufReduce2CL, true, 0, size_tables * sizeof(float), _outbuffReduce, 0, NULL, NULL);
			checkErr(ret, "Unable to read output");

			//cout << "REDUCE2_KER timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;

			/***2. Compute pseud inverse*************************************************/
			// Compute BTB

			//cout << "sizeTable: " << size_tables << endl;
			shift = 0;
			for (int i = 0; i < NB_BS-1; i++)
			{
				for (int j = i; j < NB_BS; j++)
				{
					value = double(_outbuffReduce[shift++]);
					if (j == NB_BS-1)       // vector b
						cc(i) = value;
					else
						Q(j, i) = Q(i, j) = value;
				}
			}
			//cout << "shift: " << shift << endl;
			//cout << "Residual: " << double(_outbuffReduce[shift++]) << endl;
			//cout << Q << endl;
			/*int tmpval;
			cin >> tmpval;*/

			//determinant
			double det = Q.determinant();

			if (det == 0.0/*fabs(det) < 1e-15*/ || det != det)
			{
				if (det != det) std::cout << "qnan" << endl;
				std::cout << "det null" << endl;
				//Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
				//cout << svd.singularValues() << endl;
				cout << "_BlendshapeCoeff: " << endl;
				for (int i = 1; i < NB_BS; i++) {
					cout << _BlendshapeCoeff[i] << endl;
				}
				goto SAVE_BSCOEFF;
				return;
			}
			//current_time = clock();
			Eigen::MatrixXd Q_inv = Q.inverse();
			//cout << "Q_inv timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;

			for (int i = 0; i < NB_BS-1; i++) {
				for (int j = 0; j < NB_BS-1; j++) {
					_Qinv[i * (NB_BS-1) + j] = float(Q_inv(i, j));
				}
			}

			//current_time = clock();
			/// Compute matrix PseudoInverse*B and PseudoInverse*cc
			ret = clEnqueueNDRangeKernel(_queue[PSEUDOINV_KER], _kernels[PSEUDOINV_KER], 2, NULL, gPws, lPws, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() PSEUDOINV_KER");

			ret = clFinish(_queue[PSEUDOINV_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			ret = clEnqueueNDRangeKernel(_queue[ATC_KER], _kernels[ATC_KER], 2, NULL, Rgws1, Rlws1, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() ATC_KER");

			ret = clFinish(_queue[ATC_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			ret = clEnqueueNDRangeKernel(_queue[REDSOLVE_KER], _kernels[REDSOLVE_KER], 2, NULL, Rgws2, Rlws2, 0, NULL, NULL);
			checkErr(ret, "ComamndQueue::enqueueNDRangeKernel() REDSOLVE_KER");

			ret = clFinish(_queue[REDSOLVE_KER]);
			checkErr(ret, "ComamndQueue::Finish()");

			ret = clEnqueueReadBuffer(_queue[REDSOLVE_KER], _bufSolve2CL, true, 0, (NB_BS - 1) * sizeof(float), _outbuffResolved, 0, NULL, NULL);
			checkErr(ret, "Unable to read output");

			//ret = clFinish(_queue[REDSOLVE_KER]);
			//checkErr(ret, "ComamndQueue::Finish()");

			//cout << "SOLVEPR_KER timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;

			// Compute I - Pseudo * B
			for (int i = 0; i < NB_BS-1; i++) {
				x0(i) = double(_outbuffResolved[i]);
				lb(i) = double(0.0f - _BlendshapeCoeff[i + 1]);
				ub(i) = double(1.0f - _BlendshapeCoeff[i + 1]);
			}

			//current_time = clock();
			xres = ParallelRelaxation(Q_inv, x0, lb, ub);
			//cout << "ParallelRelaxation timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;

			iter++;
			//cout << "outer_iter: " << iter /*<< " " << xres.transpose()*/ << endl;

			/*for (int i = 1; i < 28; i++) {
				if (abs(xres(i - 1)) > 0.01) {
					converged = true;
					break;
				}
			}
			if (converged)
				break;*/

			float residual = 0.0f;
			for (int i = 1; i < NB_BS; i++) {
				residual += fabs(_BlendshapeCoeff[i] - float(xres(i - 1)));
				_BlendshapeCoeff[i] = _BlendshapeCoeff[i] + float(xres(i - 1));
				if (_BlendshapeCoeff[i] < 0.0)
					_BlendshapeCoeff[i] = 0.0;
				if (_BlendshapeCoeff[i] > 1.0)
					_BlendshapeCoeff[i] = 1.0;
			}
			converged = (iter > _max_iterPR[lvl]/* || residual < 1.0e-5*/);

		}
	}

	//cout << "FINISHED" << endl;

	/*cout << "_BlendshapeCoeff: " << endl;
	for (int i = 1; i < 28; i++) {
		cout << _BlendshapeCoeff[i] << endl;
	}*/

	/*for (int i = 0; i < NB_BS; i++) {
		_BSCoeff[i].push_back(_BlendshapeCoeff[i]);
	}

	if (_BSCoeff[0].size() > 10) {
		for (int i = 0; i < NB_BS; i++) {
			_BSCoeff[i].erase(_BSCoeff[i].begin());
		}
	}*/


SAVE_BSCOEFF:
	if (save_data) {
		ofstream  filestr;

		string filename = string(dest_name) + string("\\Animation\\BSCoeff") + to_string(_idx) + string(".txt");
		filestr.open(filename, fstream::out);
		while (!filestr.is_open()) {
			cout << "Could not open MappingList" << endl;
			return;
		}

		for (int i = 0; i < NB_BS; i++) {
			filestr << _BlendshapeCoeff[i] << " " << endl;
		}

		filestr.close();
	}
	return;
}

void HeadOffV2::EstimateBlendShapeCoefficientsGaussNewton(vector<MyMesh *> Blendshape) {

	bool converge = false;
	int iter = 0;
	int max_iter = 50;

	Eigen::VectorXd delta = Eigen::VectorXd(27);
	Eigen::VectorXd xres = Eigen::VectorXd(27);
	for (int i = 1; i < 28; i++) {
		xres(i - 1) = double(_BlendshapeCoeff[i]);
	}

	float tmpPt[3];
	float tmpNMLE[3];

	// 0. Augment all vertices of all shapes with the bump image

	/*if ((_idx_curr - 2) % 100 == 0) {
	cout << "update vertices" << endl;
	std::memcpy(_BumpSwap, _Bump, BumpWidth*BumpHeight*sizeof(float));
	std::memcpy(_MaskSwap, _Mask, BumpWidth*BumpHeight*sizeof(float));
	}*/

	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };
	size_t Rgws[2] = { 1176 * STRIDE, 1 };
	size_t Rlws[2] = { STRIDE, 1 };

	cl_int ret;/* = clEnqueueNDRangeKernel(_queue[JACOBI_KER], _kernels[JACOBI_KER], 2, NULL, gws, lws, 0, NULL, NULL);
			   checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
			   ret = clFinish(_queue[JACOBI_KER]);
			   checkErr(ret, "ComamndQueue::Finish()");*/

	//// 0,5. Compute jacobian matrix JTJ
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(27, 27);
	Eigen::MatrixXd JTJ = Eigen::MatrixXd::Zero(27, 27);

	/*ret = clSetKernelArg(_kernels[REDUCE_KER], 0, sizeof(_bufJTJCL), &_bufJTJCL);
	ret = clSetKernelArg(_kernels[REDUCE_KER], 1, sizeof(_OutBuffJTJCL), &_OutBuffJTJCL);

	ret = clEnqueueNDRangeKernel(_queue[REDUCE_KER], _kernels[REDUCE_KER], 1, NULL, Rgws, Rlws, 0, NULL, NULL);
	checkErr(ret, "REDUCE_KER::enqueueNDRangeKernel()");
	ret = clFinish(_queue[REDUCE_KER]);
	checkErr(ret, "REDUCE_KER::Finish()");*/

	int shift = 0;
	for (int i = 0; i < 27; ++i) {  //rows
		for (int j = i; j < 27; ++j)    // cols + b
		{
			double value = _outbuffJTJ[shift++];
			JTJ(j, i) = JTJ(i, j) = value;
		}
		I(i, i) = JTJ(i, i);
	}

	//////////////////// ADD LANDMARKS ///////////////////////////////////
	double weightL = 100.0;

	for (int i = 0; i < 43; i++) {
		int idx_i = Myround(Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._u*float(BumpHeight));
		int idx_j = Myround(Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._v*float(BumpWidth));

		float fact_BP = 1000.0f;
		float d = 0.0f;
		if (_Bump.at<cv::Vec4f>(idx_i, idx_j)[1] == 0.0f) {
			_Vtx[0][3 * i] = 0.0f;
			_Vtx[0][3 * i + 1] = 0.0f;
			_Vtx[0][3 * i + 2] = 0.0f;
			continue;
		}
		else {
			d = _Bump.at<cv::Vec4f>(idx_i, idx_j)[0] / fact_BP;
		}

		float tmpPt[3];
		tmpPt[0] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._x + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Nx;
		tmpPt[1] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._y + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Ny;
		tmpPt[2] = Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._z + d*Blendshape[0]->_verticesList[FACIAL_LANDMARKS[i]]._Nz;

		float ptRef[3];
		_Vtx[0][3 * i] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		_Vtx[0][3 * i + 1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		_Vtx[0][3 * i + 2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);
		ptRef[0] = _Vtx[0][3 * i]; ptRef[1] = _Vtx[0][3 * i + 1]; ptRef[2] = _Vtx[0][3 * i + 2];

		for (int k = 1; k < 28; k++) {
			tmpPt[0] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._x + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Nx;
			tmpPt[1] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._y + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Ny;
			tmpPt[2] = Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._z + d*Blendshape[k]->_verticesList[FACIAL_LANDMARKS[i]]._Nz;

			_Vtx[k][3 * i] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
			_Vtx[k][3 * i + 1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
			_Vtx[k][3 * i + 2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);
		}

		for (int k = 1; k < 28; k++) {  //rows
			for (int l = k; l < 28; l++)    // cols + b
			{
				JTJ(k - 1, l - 1) = JTJ(k - 1, l - 1) + weightL * double((_Vtx[k][3 * i] - ptRef[0])*(_Vtx[l][3 * i] - ptRef[0])
					+ (_Vtx[k][3 * i + 1] - ptRef[1])*(_Vtx[l][3 * i + 1] - ptRef[1])
					+ (_Vtx[k][3 * i + 2] - ptRef[2])*(_Vtx[l][3 * i + 2] - ptRef[2]));

				JTJ(l - 1, k - 1) = JTJ(k - 1, l - 1);
			}
			I(k - 1, k - 1) = JTJ(k - 1, k - 1);
		}
	}

	//cout << JTJ << endl;
	//return;

	Rgws[0] = 50 * STRIDE;

	ret = clSetKernelArg(_kernels[REDUCE_KER], 0, sizeof(_bufCL), &_bufCL);
	ret = clSetKernelArg(_kernels[REDUCE_KER], 1, sizeof(_OutBuffCL), &_OutBuffCL);

	float prev_sum = 0.0;
	float sum;
	float count;
	Eigen::VectorXd lambda = Eigen::VectorXd(27);
	for (int i = 0; i < 27; i++)
		lambda(i) = 1.0;
	Eigen::VectorXd prev_xres = xres;
	Eigen::VectorXd b = Eigen::VectorXd::Zero(27);

	int fact = 1;
	ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 5, sizeof(int), &fact);


	while (!converge) {

		ret = clEnqueueNDRangeKernel(_queue[BSSYSTEM_KER], _kernels[BSSYSTEM_KER], 2, NULL, gws, lws, 0, NULL, NULL);
		checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
		ret = clFinish(_queue[BSSYSTEM_KER]);
		checkErr(ret, "ComamndQueue::Finish()");

		ret = clEnqueueNDRangeKernel(_queue[REDUCE_KER], _kernels[REDUCE_KER], 1, NULL, Rgws, Rlws, 0, NULL, NULL);
		checkErr(ret, "REDUCE_KER::enqueueNDRangeKernel()");
		ret = clFinish(_queue[REDUCE_KER]);
		checkErr(ret, "REDUCE_KER::Finish()");

		for (int i = 0; i < 27; ++i) {
			b(i) = _outbuff[i];
		}
		//cout << b << endl;

		sum = float(_outbuff[27]);
		count = float(_outbuff[28]);
		sum = sum / (count);

		// Add Landmarks
		float tmpPt[3];
		float pointClose[3];
		float nmleClose[3];
		for (int i = 0; i < 43; i++) {

			float pt[3];
			pt[0] = _Vtx[0][3 * i]; pt[1] = _Vtx[0][3 * i + 1]; pt[2] = _Vtx[0][3 * i + 2];
			if (pt[0] == 0.0f && pt[1] == 0.0f && pt[2] == 0.0f)
				continue;

			float ptRef[3];
			ptRef[0] = _Vtx[0][3 * i]; ptRef[1] = _Vtx[0][3 * i + 1]; ptRef[2] = _Vtx[0][3 * i + 2];

			for (int k = 1; k < 28; k++) {
				pt[0] = pt[0] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i] - ptRef[0]);
				pt[1] = pt[1] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 1] - ptRef[1]);
				pt[2] = pt[2] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 2] - ptRef[2]);
			}
			pt[0] = pt[0] + _Translation_inv(0);
			pt[1] = pt[1] + _Translation_inv(1);
			pt[2] = pt[2] + _Translation_inv(2);

			// 2. Compute associated points
			float min_dist;
			tmpPt[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0];
			tmpPt[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1];
			tmpPt[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2];

			//tmpNMLE[0] = _NMap.at<cv::Vec3f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0];
			//tmpNMLE[1] = _NMap.at<cv::Vec3f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1];
			//tmpNMLE[2] = _NMap.at<cv::Vec3f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2];
			if (tmpPt[0] == 0.0 && tmpPt[1] == 0.0 && tmpPt[2] == 0.0f)
				min_dist = 1000.0f;

			min_dist = sqrt((tmpPt[0] - pt[0])*(tmpPt[0] - pt[0]) + (tmpPt[1] - pt[1])*(tmpPt[1] - pt[1]) + (tmpPt[2] - pt[2])*(tmpPt[2] - pt[2]));

			if (min_dist < 10.1f) {
				for (int k = 1; k < 28; k++)    // cols + b
				{
					b(k - 1) = b(k - 1) + weightL * double((_Vtx[k][3 * i] - ptRef[0])*(tmpPt[0] - pt[0])
						+ (_Vtx[k][3 * i + 1] - ptRef[1])*(tmpPt[1] - pt[1])
						+ (_Vtx[k][3 * i + 2] - ptRef[2])*(tmpPt[2] - pt[2]));
				}
			}
		}

		//cout << "sum: " << sum << "prev_sum: " << prev_sum << endl;
		for (int k = 0; k < 27; k++)
			I(k, k) = JTJ(k, k) * 100.0 / double(1.0f + 100000.0f*fabs(sum - prev_sum)); // lambda(k);

		//cout << "xres: " << xres << "prev_xres: " << prev_xres << endl;
		if (prev_sum != 0.0f) {
			if (sum > prev_sum) {
				xres = prev_xres;
				lambda = lambda / 1.5;
				iter++;
				converge = iter > max_iter;
				for (int i = 1; i < 28; i++)
					_BlendshapeCoeff[i] = float(xres(i - 1));
				continue;
			}
			else {
				if (sum < prev_sum) {
					prev_xres = xres;
					prev_sum = sum;
				}
			}
		}
		else {
			prev_sum = sum;
			prev_xres = xres;
		}

		//cout << "lambda: " << lambda << "prev_sum: " << prev_sum << endl;

		//cout << "sum: " << sum << endl;
		//cout << "count: " << count << endl;

		delta = (JTJ + I).llt().solve(b);
		for (int k = 0; k < 27; k++)
			xres(k) = xres(k) + lambda(k)*delta(k);

		for (int k = 0; k < 27; k++) {
			if (xres(k) < 0.0 || xres(k) > 1.0) {
				xres(k) = prev_xres(k);
				lambda(k) = lambda(k) / 1.5;
			}
		}

		for (int i = 1; i < 28; i++)
			_BlendshapeCoeff[i] = float(xres(i - 1));
		//cout << "xres: " << xres << endl;

		iter++;
		converge = iter > max_iter;
	}

	for (int i = 1; i < 28; i++) {
		_BlendshapeCoeff[i] = float(xres(i - 1));
		cout << _BlendshapeCoeff[i] << endl;
	}

	//int tmp;
	//cin >> tmp;


	// Compute Vertex map
	/*ret = clEnqueueNDRangeKernel(_queue[ANIMATE_KER], _kernels[ANIMATE_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[ANIMATE_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");*/

}

void HeadOffV2::SetLandmarks(MyMesh *RefMesh) {

	_landmarkOK = true;

	float ptLM[3];
	//for (int i = 31; i < 43; i++) {
	//	if (_landmarks.cols < i + 1)
	//		continue;

	//	ptLM[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0];
	//	ptLM[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1];
	//	ptLM[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2];

	//	if (ptLM[0] == 0.0f && ptLM[1] == 0.0f && ptLM[2] == 0.0f) {
	//		cout << "No landmark VMAP!" << endl;
	//		continue;
	//	}

	//	// Search for closest point in the bump image
	//	float min_dist = 1.0e6;
	//	int best_i, best_j;
	//	float pt[3];
	//	for (int u = 0; u < BumpHeight; u ++) {
	//		for (int v = 0; v < BumpWidth; v ++) {
	//			if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f)
	//				continue;

	//			pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
	//			pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
	//			pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

	//			/*pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0];
	//			pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[1];
	//			pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[2];*/
	//			float dist = sqrt((pt[0] - ptLM[0])*(pt[0] - ptLM[0]) + (pt[1] - ptLM[1])*(pt[1] - ptLM[1]) + (pt[2] - ptLM[2])*(pt[2] - ptLM[2]));
	//			if (dist < min_dist) {
	//				min_dist = dist;
	//				best_i = u;
	//				best_j = v;
	//			}
	//		}
	//	}

	//	if (min_dist < 0.005) {
	//		_landmarksBump.at<int>(0, i) = best_i;
	//		_landmarksBump.at<int>(1, i) = best_j;
	//		//cout << "best_i: " << best_i << "; best_j: " << best_j << endl;
	//	}
	//	else {
	//		_landmarksBump.at<int>(0, i) = -1;
	//		_landmarksBump.at<int>(1, i) = -1;
	//		//cout << "No landmark!" << endl;
	//		_landmarkOK = false;
	//	}

	//}

	for (int i = 0; i < 43; i++) {
		int best_i = Myround(RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_u*float(BumpHeight));
		int best_j = Myround(RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_v*float(BumpWidth));
		if (i == 22) {
			best_i = 110;
			best_j = 169;
		}
		if (i == 25) {
			best_i = 130;
			best_j = 169;
		}

		/*if (i == 38) {
			best_i = 128;
			best_j = 122;
		}

		if (i == 39) {
			best_i = 126;
			best_j = 122;
		}

		if (i == 40) {
			best_i = 120;
			best_j = 122;
		}

		if (i == 41) {
			best_i = 114;
			best_j = 122;
		}

		if (i == 42) {
			best_i = 110;
			best_j = 122;
		}*/

		_landmarksBump.at<int>(0, i) = best_i;
		_landmarksBump.at<int>(1, i) = best_j;
	}
}

void HeadOffV2::LoadAnimatedModel() {
	// Load the Bump and RGB images.
	float bumpval;
	cv::Mat img = cv::imread(string("Bump.png"), CV_LOAD_IMAGE_UNCHANGED);
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			if (img.at<cv::Vec3b>(i, j)[0] == 255 && img.at<cv::Vec3b>(i, j)[1] == 255 && img.at<cv::Vec3b>(i, j)[2] == 255) {
				_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_Bump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				continue;
			}

			if (img.at<cv::Vec3b>(i, j)[2] == 0) {
				//bumpval = ((((float(img.at<cv::Vec3b>(i, j)[1]) / 2.0f) / 255.0f)*2000.0f) - 1000.0f) / 50.0f;
				bumpval = ((((float(img.at<cv::Vec3b>(i, j)[1]) / 2.0f) / 255.0f)*6000.0f) - 3000.0f) / 50.0f;
			}
			else {
				//bumpval = (((((float(img.at<cv::Vec3b>(i, j)[2]) / 2.0f) + 128.0f) / 255.0f)*2000.0f) - 1000.0f) / 50.0f;
				bumpval = (((((float(img.at<cv::Vec3b>(i, j)[2]) / 2.0f) + 128.0f) / 255.0f)*6000.0f) - 3000.0f) / 50.0f;
			}
			_Bump.at<cv::Vec4f>(i, j)[0] = bumpval;
			_Bump.at<cv::Vec4f>(i, j)[1] = 1.0f;
		}
	}

	//Load RGB image
	img = cv::imread(string("BumpRGB.png"), CV_LOAD_IMAGE_UNCHANGED);
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			_RGBMapBump.at<cv::Vec4f>(i, j)[2] = float(img.at<cv::Vec3b>(i, j)[0]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[1] = float(img.at<cv::Vec3b>(i, j)[1]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[0] = float(img.at<cv::Vec3b>(i, j)[2]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[3] = 1.0f;
		}
	}

	cl_int ret;
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _BumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _Bump.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _LabelsMaskCL, true, origin, region, BumpWidth * 4 * sizeof(unsigned char), 0, _LabelsMask.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");
	ret = clEnqueueWriteImage(_queue[BUMP_KER], _WeightMapCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _WeightMap.data, 0, NULL, NULL);
	checkErr(ret, "Unable to read output");


	/**********************************Initialise Bump attributes*******************************************/
	float colfill[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	ret = clEnqueueFillImage(_queue[BUMP_KER], _VMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage VMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _NMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage NMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _RGBMapBumpCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage RGBMapBumpCL");
	ret = clEnqueueFillImage(_queue[BUMP_KER], _RGBMapBumpSwapCL, colfill, origin, region, 0, NULL, NULL);
	checkErr(ret, "FillImage RGBMapBumpCL");

	// Compute HD Blendshape Vertices 
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	ret = clEnqueueNDRangeKernel(_queue[DATAPROC_KER], _kernels[DATAPROC_KER], 2, NULL, gws, lws, 0, NULL, NULL);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");
	ret = clFinish(_queue[DATAPROC_KER]);
	checkErr(ret, "ComamndQueue::Finish()");
}

void HeadOffV2::AnimateModel() {
	cl_event evts[3];
	cl_int ret;
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { BumpWidth, BumpHeight, 1 };

	// Compute Vertex map
	int gws_x = divUp(BumpHeight, THREAD_SIZE_X);
	int gws_y = divUp(BumpWidth, THREAD_SIZE_Y);
	size_t gws[2] = { gws_x*THREAD_SIZE_X, gws_y*THREAD_SIZE_Y };
	size_t lws[2] = { THREAD_SIZE_X, THREAD_SIZE_Y };

	ret = clEnqueueNDRangeKernel(_queue[VMAPBUMP_KER], _kernels[VMAPBUMP_KER], 2, NULL, gws, lws, 0, NULL, &evts[0]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clFinish(_queue[VMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

	ret = clEnqueueReadImage(_queue[VMAPBUMP_KER], _VMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _VMapBump.data, 1, &evts[0], NULL);
	checkErr(ret, "Unable to read VMAPBUMP_KER output");

	ret = clEnqueueNDRangeKernel(_queue[NMAPBUMP_KER], _kernels[NMAPBUMP_KER], 2, NULL, gws, lws, 1, &evts[0], &evts[1]);
	checkErr(ret, "ComamndQueue::enqueueNDRangeKernel()");

	ret = clEnqueueReadImage(_queue[NMAPBUMP_KER], _NMapBumpCL, true, origin, region, BumpWidth * 4 * sizeof(float), 0, _NMapBump.data, 1, &evts[1], NULL);
	checkErr(ret, "Unable to read NMAPBUMP_KER output");

	ret = clFinish(_queue[NMAPBUMP_KER]);
	checkErr(ret, "ComamndQueue::Finish()");

}

void HeadOffV2::LoadCoeffPose(char *path) {
	ifstream  filestr;
	char tmpline[256];

	string filename = string(path) + string("\\BSCoeff") + to_string(_idx) + string(".txt");
	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		_idx++;
		return;
	}

	for (int i = 0; i < NB_BS; i++) {
		filestr.getline(tmpline, 256);
		float tmpval;
		sscanf_s(tmpline, "%f", &tmpval);
		_BlendshapeCoeff[i] = tmpval;
		//cout << _BlendshapeCoeff[i] << endl;
	}

	filestr.close();

	filename = string(path) + string("\\Pose") + to_string(_idx) + string(".txt");
	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		_idx++;
		return;
	}

	for (int i = 0; i < 16; i++) {
		filestr.getline(tmpline, 256);
		sscanf_s(tmpline, "%f", &_Pose[i]);
	}

	filestr.close();
	_idx++;
	_idx_curr++;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Rotation_inv(j, i) = _Pose[4 * i + j];
		}
		_Translation_inv(i) = _Pose[12 + i];
	}

	cout << "Param loaded" << endl;
}