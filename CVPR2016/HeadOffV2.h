#ifndef __HEADOFFV2_H
#define __HEADOFFV2_H

#include "KinectCapture.h"

cl_kernel LoadKernel(string filename, string Kernelname, cl_context context, cl_device_id device);

class HeadOffV2 {

public:
	cl_context _context;
	cl_device_id _device;
	std::vector<cl_kernel> _kernels;
	std::vector<cl_command_queue> _queue;
	int _idx_curr;

	bool _Kinect1;
	bool _Kinect2;
	int _KinectVersion;
	bool _landmarkOK;

	cv::Mat _depth_in[2];
	cv::Mat _color_in[2];
	int _idx_thread[2];

	//Class to read from Kinect
	SkeletonTrack* _Ske;

	//Class to read from KinectV2
	KinectV2Manager* _KinectV2;

	int *TABLE_I;
	int *TABLE_J;

	////////////// Blendshape memory ////////////////

	FaceGPU * _triangles;
	Point3DGPU * _verticesList;

	////////////////////////////////////////////////////

	HeadOffV2(cl_context context, cl_device_id device) : _context(context), _device(device), _draw(false), _idx(0), _idx_curr(0), _lvl(1), _landmarkOK(false) {
		// set up face tracker
		_sdm = unique_ptr<SDM>(new SDM("..\\..\\models\\tracker_model49_131.bin"));
		_hpe = unique_ptr<HPE>(new HPE());
		_restart = true;
		_minScore = 0.1f; 

		_max_iter[0] = 6;
		_max_iter[1] = 0;
		_max_iter[2] = 0;

		_max_iterPR[0] = 6;
		_max_iterPR[1] = 0;
		_max_iterPR[2] = 0;

		//int size_tables = ((NB_BS - 1)*(NB_BS - 2)) / 2 + 2*(NB_BS - 1) + 1; // +1 to compute residual error
		int size_tables = ((NB_BS)*(NB_BS + 1)) / 2;

		TABLE_I = (int *)malloc(size_tables*sizeof(int));
		TABLE_J = (int *)malloc(size_tables*sizeof(int));
		_Qinv = (float *)malloc((NB_BS - 1)*(NB_BS - 1)*sizeof(float));

		int indx = 0;
		for (int i = 0; i < NB_BS; i++) {
			for (int j = i; j < NB_BS; j++) {
				TABLE_I[indx] = i;
				TABLE_J[indx] = j;
				//cout << TABLE_I[indx] << " ";
				indx++;
			}
			//cout << " " << endl;
		}


		cv::Mat imgL = cv::imread(string("LabelsMask.bmp"), CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat img0 = cv::imread(string("Weights-240.png"), CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat img1 = cv::imread(string("Labels-240.png"), CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat img2 = cv::imread(string("FrontFace.png"), CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat img3 = cv::imread(string("Labelsb-240.png"), CV_LOAD_IMAGE_UNCHANGED);

		_vertices = (MyPoint **)malloc(cDepthHeight*cDepthWidth*sizeof(MyPoint *));
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				_vertices[i*cDepthWidth + j] = NULL;
			}
		}

		for (int k = 0; k < 28; k++) {
			_Vtx[k] = (float *)malloc(3 * NBVertices*sizeof(float));
		}
		_imgD = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);
		_imgC = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
		_imgS = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);


		_depth_in[0] = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);
		_color_in[0] = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
		_depth_in[1] = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);
		_color_in[1] = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);

		_depthIn = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC3);
		_VMap = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);
		_NMap = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);
		_RGBMap = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
		_segmentedIn = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

		_VMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
		_NMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
		_RGBMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);

		_WeightMap = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
		_Bump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
		_BumpSwap = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
		_LabelsMask = cv::Mat(BumpHeight, BumpWidth, CV_8UC4);

		_triangles = (FaceGPU *)malloc(8518 * sizeof(FaceGPU));
		_verticesList = (Point3DGPU *)malloc(49 * 4325 * sizeof(Point3DGPU));
		
		_verticesBump = (Point3DGPU *)malloc(BumpHeight * BumpWidth*sizeof(Point3DGPU));
		for (int i = 0; i < BumpHeight; i++) {
			for (int j = 0; j < BumpWidth; j++) {
				_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_Bump.at<cv::Vec4f>(i, j)[1] = -200.0f; // Mask   for VMP -200, otherwise 0
				_Bump.at<cv::Vec4f>(i, j)[2] = float(img1.at<cv::Vec3w>(i, j)[0]) - 1.0f;
				_Bump.at<cv::Vec4f>(i, j)[3] = img3.at<cv::Vec4b>(i, j)[2] > 100 ? -1.0f : 0.0f;
				_LabelsMask.at<cv::Vec4b>(i, j)[0] = imgL.at<cv::Vec3b>(i, j)[0];
				_LabelsMask.at<cv::Vec4b>(i, j)[1] = imgL.at<cv::Vec3b>(i, j)[1];
				_LabelsMask.at<cv::Vec4b>(i, j)[2] = imgL.at<cv::Vec3b>(i, j)[2];
				_LabelsMask.at<cv::Vec4b>(i, j)[3] = img2.at<cv::Vec4b>(i, j)[2] > 100 ? 1 : 0;
				_WeightMap.at<cv::Vec4f>(i, j)[0] = float(img0.at<cv::Vec3w>(i, j)[0]) / 65535.0f;
				_WeightMap.at<cv::Vec4f>(i, j)[1] = float(img0.at<cv::Vec3w>(i, j)[1]) / 65535.0f;
				_WeightMap.at<cv::Vec4f>(i, j)[2] = float(img0.at<cv::Vec3w>(i, j)[2]) / 65535.0f;
			}
		}

		for (int i = 0; i < 16; i++)
			_Pose[i] = 0.0;
		_Pose[0] = 1.0; _Pose[5] = 1.0; _Pose[10] = 1.0; _Pose[15] = 1.0;

		_outbuff = (double *)malloc(50 * sizeof(double));
		_outbuffJTJ = (double *)malloc(1176 * sizeof(double));
		_outbuffGICP = (double *)malloc(29 * sizeof(double));
		_outbuffReduce = (float *)malloc(size_tables * sizeof(float));
		_outbuffResolved = (float *)malloc((NB_BS-1) * sizeof(float));

		_landmarksBump = cv::Mat(2, 43, CV_32SC1);
		_landmarks = cv::Mat(2, 43, CV_32FC1);

		for (int i = 0; i < 43; i++) {
			_landmarks.at<float>(0, i) = 0.0f;
			_landmarks.at<float>(1, i) = 0.0f;
		}

		////////// OPENCL memory ////////////
		cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT16 };
		cl_image_format format2 = { CL_RGBA, CL_UNSIGNED_INT8 };
		cl_image_format format3 = { CL_RGBA, CL_FLOAT };
		cl_image_desc  desc;
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = cDepthWidth;
		desc.image_height = cDepthHeight;
		desc.image_depth = 0;
		desc.image_array_size = 0;
		desc.image_row_pitch = 0;
		desc.image_slice_pitch = 0;
		desc.num_mip_levels = 0;
		desc.num_samples = 0;
		desc.buffer = NULL;

		cl_image_desc  desc2;
		desc2.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc2.image_width = BumpWidth;
		desc2.image_height = BumpHeight;
		desc2.image_depth = 0;
		desc2.image_array_size = 0;
		desc2.image_row_pitch = 0;
		desc2.image_slice_pitch = 0;
		desc2.num_mip_levels = 0;
		desc2.num_samples = 0;
		desc2.buffer = NULL;

		/*cout << CL_DEVICE_IMAGE3D_MAX_WIDTH << endl;
		cout << CL_DEVICE_IMAGE2D_MAX_HEIGHT << endl;
		cout << CL_DEVICE_IMAGE3D_MAX_DEPTH << endl;


		cl_image_desc  desc3;
		desc2.image_type = CL_MEM_OBJECT_IMAGE3D;
		desc2.image_width = BumpWidth;
		desc2.image_height = BumpHeight;
		desc2.image_depth = 49;
		desc2.image_array_size = 0;
		desc2.image_row_pitch = 0;
		desc2.image_slice_pitch = 0;
		desc2.num_mip_levels = 0;
		desc2.num_samples = 0;
		desc2.buffer = NULL;*/

		cl_int ret;
		_depthCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format, &desc, NULL, &ret);
		checkErr(ret, "_depthCL::Buffer()");
		_depthBuffCL = clCreateImage(_context, CL_MEM_READ_ONLY, &format, &desc, NULL, &ret);
		checkErr(ret, "_depthBuffCL::Buffer()");
		_VMapCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc, NULL, &ret);
		checkErr(ret, "_VMapCL::Buffer()");
		_NMapCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc, NULL, &ret);
		checkErr(ret, "_NMapCL::Buffer()");
		_RGBMapCL = clCreateImage(_context, CL_MEM_READ_ONLY, &format2, &desc, NULL, &ret);
		checkErr(ret, "_RGBMapCL::Buffer()");

		_SegmentedCL = clCreateImage(_context, CL_MEM_READ_ONLY, &format2, &desc, NULL, &ret);
		checkErr(ret, "_SegmentedCL::Buffer()");

		_intrinsicCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 11 * sizeof(float), _intrinsic, &ret);
		checkErr(ret, "_intrinsics::Buffer()");
		_PoseCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 16 * sizeof(float), _Pose, &ret);
		checkErr(ret, "_PoseCL::Buffer()");
		_QINVCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, NB_BS * NB_BS * sizeof(float), _Qinv, &ret);
		checkErr(ret, "_QINVCL::Buffer()");
		_B = clCreateBuffer(_context, CL_MEM_READ_WRITE, NB_BS * (43 * 3 + BumpHeight*BumpWidth) * sizeof(float), NULL, &ret);
		checkErr(ret, "_B::Buffer()");
		_PseudoInverseCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, NB_BS * (43 * 3 + BumpHeight*BumpWidth) * sizeof(float), NULL, &ret);
		checkErr(ret, "_PseudoInverseCL::Buffer()");
		

		_BlendshapeCoeffCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NB_BS * sizeof(float), _BlendshapeCoeff, &ret);
		checkErr(ret, "_BlendshapeCoeffCL::Buffer()");

		_VerticesCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NB_BS * 4325 * sizeof(Point3DGPU), _verticesList, &ret);
		checkErr(ret, "_VerticesCL::Buffer()");
		_TrianglesCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 8518 * sizeof(FaceGPU), _triangles, &ret);
		checkErr(ret, "_TrianglesCL::Buffer()");

		_WeightMapCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_WeightMapCL::Buffer()");
		_LabelsMaskCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format2, &desc2, NULL, &ret);
		checkErr(ret, "_LabelsMaskCL::Buffer()");

		_BumpCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_BumpCL::Buffer()");
		_BumpSwapCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_BumpSwapCL::Buffer()");

		//_verticesBumpCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BumpHeight * BumpWidth*sizeof(Point3DGPU), _verticesBump, &ret);
		_VMapBumpCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_VMapBumpCL::Buffer()");
		_NMapBumpCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_NMapBumpCL::Buffer()");
		_RGBMapBumpCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_RGBMapBumpCL::Buffer()");
		_RGBMapBumpSwapCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc2, NULL, &ret);
		checkErr(ret, "_RGBMapBumpCL::Buffer()");

		//_verticesBSCL = clCreateImage(_context, CL_MEM_READ_WRITE, &format3, &desc3, NULL, &ret);
		_verticesBSCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, NB_BS * 6 * BumpHeight * BumpWidth*sizeof(float), NULL, &ret);
		checkErr(ret, "_verticesBSCL::Buffer()");

		_LandMarksBumpCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 43 * 2 *sizeof(int), _landmarksBump.data, &ret);
		checkErr(ret, "_LandMarksBumpCL::Buffer()"); 
		_LandMarksCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 43 * 2 * sizeof(float), _landmarks.data, &ret);
		checkErr(ret, "_LandMarksCL::Buffer()");

		int dim_x = divUp(BumpHeight, THREAD_SIZE_X);
		int dim_y = divUp(BumpWidth, THREAD_SIZE_Y);
		_bufCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, 50 * dim_x*dim_y * sizeof(double), NULL, &ret);
		checkErr(ret, "_bufCL::Buffer()");
		_OutBuffCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 50 * sizeof(double), _outbuff, &ret);
		checkErr(ret, "_OutBuffCL::Buffer()");
		_bufJTJCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, 1176 * dim_x*dim_y * sizeof(double), NULL, &ret);
		checkErr(ret, "_bufJTJCL::Buffer()");
		_OutBuffJTJCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1176 * sizeof(double), _outbuffJTJ, &ret);
		checkErr(ret, "_OutBuffJTJCL::Buffer()");

		_bufGICPCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, 29 * dim_x*dim_y * sizeof(double), NULL, &ret);
		checkErr(ret, "_bufCL::Buffer()");
		_OutBuffGICPCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 27 * sizeof(double), _outbuffGICP, &ret);
		checkErr(ret, "_OutBuffCL::Buffer()");

		int dim = divUp(3 * 43 + BumpHeight*BumpWidth, (STRIDE * 2));
		_bufReduce1CL = clCreateBuffer(_context, CL_MEM_READ_WRITE, size_tables * dim * sizeof(float), NULL, &ret);
		checkErr(ret, "_bufCL::Buffer()");
		_bufReduce2CL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_tables * sizeof(float), _outbuffReduce, &ret);
		checkErr(ret, "_OutBuffCL::Buffer()");

		_bufSolve1CL = clCreateBuffer(_context, CL_MEM_READ_WRITE, NB_BS * dim * sizeof(float), NULL, &ret);
		checkErr(ret, "_bufCL::Buffer()");
		_bufSolve2CL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (NB_BS-1) * sizeof(float), _outbuffResolved, &ret);
		checkErr(ret, "_OutBuffCL::Buffer()");


		_TableiCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_tables * sizeof(int), (int *)TABLE_I, &ret);
		checkErr(ret, "_TableiCL::Buffer()");
		_TablejCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_tables * sizeof(int), (int *)TABLE_J, &ret);
		checkErr(ret, "_TablejCL::Buffer()");

		_TableGICPiCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 28 * sizeof(int), (int *)TABLEGICP_I, &ret); // Should be 27!!
		checkErr(ret, "_TableiCL::Buffer()");
		_TableGICPjCL = clCreateBuffer(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 28 * sizeof(int), (int *)TABLEGICP_J, &ret);
		checkErr(ret, "_TablejCL::Buffer()");

		_NbMatchesCL = clCreateBuffer(_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int), &_nbMatches, &ret);
		checkErr(ret, "_TablejCL::Buffer()");
		

		////////////////////// VMP data allocation //////

		float intbuff[BumpWidth*BumpHeight * 2];
		for (int id = 0; id < BumpWidth*BumpHeight; id++) {
			intbuff[2 * id] = 0.0f;
			intbuff[2 * id + 1] = 0.0f;
		}

		_NaturalParamCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BumpWidth*BumpHeight * 2 * sizeof(float), intbuff, &ret);
		checkErr(ret, "_NaturalParamCL::Buffer()");

		for (int id = 0; id < BumpWidth*BumpHeight; id ++) {
			intbuff[2 * id] = 0.0f;
			intbuff[2 * id + 1] = -200.0f;
		}

		_PriorCL = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BumpWidth*BumpHeight * 2 * sizeof(float), intbuff, &ret);
		checkErr(ret, "_PriorCL::Buffer()");
		delete intbuff;

		_ParentsCL = clCreateBuffer(_context, CL_MEM_READ_WRITE , cDepthWidth*cDepthHeight *100*sizeof(int), NULL, &ret);
		checkErr(ret, "_ParentsCL::Buffer()");

		_ChildsCL = clCreateBuffer(_context, CL_MEM_READ_WRITE, BumpWidth*BumpHeight * 100 * sizeof(int), NULL, &ret);
		checkErr(ret, "_ChildsCL::Buffer()");

		/////////////////////////////////////

		_pRect.x = 0;
		_pRect.y = 0;
		_pRect.width = cDepthWidth;
		_pRect.height = cDepthHeight;

		for (int i = 0; i < 49; i++)
			_BlendshapeCoeff[i] = 0.0;
		_BlendshapeCoeff[0] = 1.0;

		_Rotation = Eigen::Matrix3f::Identity();
		_Translation = Eigen::Vector3f::Zero();
		_Rotation_inv = Eigen::Matrix3f::Identity();
		_Translation_inv = Eigen::Vector3f::Zero();


		////////// LOAD OPENCL KERNELS ////////////////////////////
		int Kversion = 0;
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\VMap.cl"), string("VmapKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[VMAP_KER], 0, sizeof(_depthCL), &_depthCL);
		ret = clSetKernelArg(_kernels[VMAP_KER], 1, sizeof(_SegmentedCL), &_SegmentedCL);
		ret = clSetKernelArg(_kernels[VMAP_KER], 2, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[VMAP_KER], 3, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[VMAP_KER], 4, sizeof(int), &Kversion);
		ret = clSetKernelArg(_kernels[VMAP_KER], 5, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[VMAP_KER], 6, sizeof(cDepthWidth), &cDepthWidth);
		checkErr(ret, "kernelVMap::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Nmap.cl"), string("NmapKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[NMAP_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[NMAP_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[NMAP_KER], 2, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[NMAP_KER], 3, sizeof(cDepthWidth), &cDepthWidth);
		checkErr(ret, "kernelNMAP_KER::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Bump2.cl"), string("BumpKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[BUMP_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 2, sizeof(_RGBMapCL), &_RGBMapCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 3, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 4, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 5, sizeof(_RGBMapBumpCL), &_RGBMapBumpCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 6, sizeof(_RGBMapBumpSwapCL), &_RGBMapBumpSwapCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 7, sizeof(_VMapBumpCL), &_VMapBumpCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 8, sizeof(_NMapBumpCL), &_NMapBumpCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 9, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 10, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 11, sizeof(_LabelsMaskCL), &_LabelsMaskCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 12, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 13, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[BUMP_KER], 14, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[BUMP_KER], 15, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[BUMP_KER], 16, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[BUMP_KER], 17, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "BumpKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\NmapBump.cl"), string("NmapBumpKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[NMAPBUMP_KER], 0, sizeof(_VMapBumpCL), &_VMapBumpCL);
		ret = clSetKernelArg(_kernels[NMAPBUMP_KER], 1, sizeof(_NMapBumpCL), &_NMapBumpCL);
		ret = clSetKernelArg(_kernels[NMAPBUMP_KER], 2, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[NMAPBUMP_KER], 3, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "NMapBumpKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\VMapBump.cl"), string("VmapBumpKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 0, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 1, sizeof(_VMapBumpCL), &_VMapBumpCL);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 2, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 3, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 4, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 5, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[VMAPBUMP_KER], 6, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "VmapBumpKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\DenseBS.cl"), string("DenseBSKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 0, sizeof(_BumpCL), &_BumpCL);
		//ret = clSetKernelArg(_kernels[DENSEBS_KER], 1, sizeof(_MaskCL), &_MaskCL);
		//ret = clSetKernelArg(_kernels[DENSEBS_KER], 2, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 3, sizeof(_VerticesCL), &_VerticesCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 4, sizeof(_WeightMapCL), &_WeightMapCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 5, sizeof(_TrianglesCL), &_TrianglesCL);
		//ret = clSetKernelArg(_kernels[DENSEBS_KER], 6, sizeof(_LabelCL), &_LabelCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 7, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 8, sizeof(_bufJTJCL), &_bufJTJCL);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 9, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 10, sizeof(BumpWidth), &BumpWidth);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 11, sizeof(dim_x), &dim_x);
		ret = clSetKernelArg(_kernels[DENSEBS_KER], 12, sizeof(dim_y), &dim_y);
		checkErr(ret, "DenseBSKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\System.cl"), string("SystemKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 2, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 3, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 4, sizeof(_bufCL), &_bufCL);

		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 6, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 7, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 8, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 9, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 10, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 11, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 12, sizeof(BumpWidth), &BumpWidth);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 13, sizeof(dim_x), &dim_x);
		ret = clSetKernelArg(_kernels[BSSYSTEM_KER], 14, sizeof(dim_y), &dim_y);
		checkErr(ret, "SystemKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		int length = dim_x*dim_y;
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Reduce.cl"), string("ReduceKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[REDUCE_KER], 0, sizeof(_bufJTJCL), &_bufJTJCL);
		ret = clSetKernelArg(_kernels[REDUCE_KER], 1, sizeof(_OutBuffJTJCL), &_OutBuffJTJCL);
		ret = clSetKernelArg(_kernels[REDUCE_KER], 2, sizeof(length), &length);
		checkErr(ret, "ReduceKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\DataProc.cl"), string("DataProcKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 0, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 1, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 2, sizeof(_VerticesCL), &_VerticesCL);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 3, sizeof(_WeightMapCL), &_WeightMapCL);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 4, sizeof(_TrianglesCL), &_TrianglesCL);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 5, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[DATAPROC_KER], 6, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "DataProcKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");


		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\GICPB.cl"), string("GICPKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[GICP_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 2, sizeof(_VMapBumpCL), &_VMapBumpCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 3, sizeof(_NMapBumpCL), &_NMapBumpCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 4, sizeof(_B), &_B);
		ret = clSetKernelArg(_kernels[GICP_KER], 5, sizeof(_LabelsMaskCL), &_LabelsMaskCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 7, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 8, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 9, sizeof(_NbMatchesCL), &_NbMatchesCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 10, sizeof(_LandMarksBumpCL), &_LandMarksBumpCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 11, sizeof(_LandMarksCL), &_LandMarksCL);
		ret = clSetKernelArg(_kernels[GICP_KER], 12, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[GICP_KER], 13, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[GICP_KER], 14, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[GICP_KER], 15, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "GICPKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		int nb_lines = 8;
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\BTB.cl"), string("BTBKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 0, sizeof(_B), &_B);
		ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 1, sizeof(_bufReduce1CL), &_bufReduce1CL);
		ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 2, sizeof(_TableGICPiCL), &_TableGICPiCL);
		ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 3, sizeof(_TableGICPjCL), &_TableGICPjCL);
		ret = clSetKernelArg(_kernels[REDUCEGICP_KER], 4, sizeof(nb_lines), &nb_lines);
		checkErr(ret, "ReduceKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Jacob.cl"), string("JacobiKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[JACOBI_KER], 0, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 1, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 2, sizeof(_bufJTJCL), &_bufJTJCL);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 3, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 4, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 5, sizeof(BumpWidth), &BumpWidth);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 6, sizeof(dim_x), &dim_x);
		ret = clSetKernelArg(_kernels[JACOBI_KER], 7, sizeof(dim_y), &dim_y);
		checkErr(ret, "JacobiKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");


		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\SystemPR.cl"), string("SystemPRKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 2, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 3, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 4, sizeof(_bufJTJCL), &_bufJTJCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 6, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 7, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 8, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 9, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 10, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 11, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 12, sizeof(BumpWidth), &BumpWidth);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 13, sizeof(dim_x), &dim_x);
		ret = clSetKernelArg(_kernels[SYSTEMPR_KER], 14, sizeof(dim_y), &dim_y);
		checkErr(ret, "SystemPR::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\SolveInitPb.cl"), string("SolveInitPbKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 2, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 3, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 4, sizeof(_bufJTJCL), &_bufJTJCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 6, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 7, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 8, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 9, sizeof(_QINVCL), &_QINVCL);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 10, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 11, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 12, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 13, sizeof(BumpWidth), &BumpWidth);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 14, sizeof(dim_x), &dim_x);
		ret = clSetKernelArg(_kernels[SOLVEPR_KER], 15, sizeof(dim_y), &dim_y);
		checkErr(ret, "SolveInitPb::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\SystemPRb.cl"), string("SystemPRbKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 2, sizeof(_NMapBumpCL), &_NMapBumpCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 3, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 4, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 5, sizeof(_B), &_B);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 7, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 8, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 9, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 10, sizeof(_LandMarksBumpCL), &_LandMarksBumpCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 11, sizeof(_LandMarksCL), &_LandMarksCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 12, sizeof(_LabelsMaskCL), &_LabelsMaskCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 13, sizeof(_NbMatchesCL), &_NbMatchesCL);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 14, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 15, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 16, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[SYSTEMPRB_KER], 17, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "SystemPRB::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		int nb_lines2 = NB_BS;
		length = 3*43 + BumpHeight*BumpWidth;
		int length_out = divUp(length, (STRIDE * 2));
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\BTB.cl"), string("BTBKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 0, sizeof(_B), &_B);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 1, sizeof(_bufReduce1CL), &_bufReduce1CL);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 2, sizeof(_TableiCL), &_TableiCL);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 3, sizeof(_TablejCL), &_TablejCL);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 4, sizeof(nb_lines2), &nb_lines2);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 5, sizeof(length), &length);
		ret = clSetKernelArg(_kernels[REDUCE1_KER], 6, sizeof(length_out), &length_out);
		checkErr(ret, "Reduce1Kernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		length = divUp(3 * 43 + BumpHeight*BumpWidth, (STRIDE * 2));
		length_out = divUp(length, (STRIDE * 2));
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Reduce2.cl"), string("Reduce2Kernel"), _context, _device));
		ret = clSetKernelArg(_kernels[REDUCE2_KER], 0, sizeof(_bufReduce1CL), &_bufReduce1CL);
		ret = clSetKernelArg(_kernels[REDUCE2_KER], 1, sizeof(_bufReduce2CL), &_bufReduce2CL);
		ret = clSetKernelArg(_kernels[REDUCE2_KER], 2, sizeof(length), &length);
		ret = clSetKernelArg(_kernels[REDUCE2_KER], 3, sizeof(length_out), &length_out);
		checkErr(ret, "Reduce2Kernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\PseudoInverse.cl"), string("PseudoInverseKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[PSEUDOINV_KER], 0, sizeof(_PseudoInverseCL), &_PseudoInverseCL);
		ret = clSetKernelArg(_kernels[PSEUDOINV_KER], 2, sizeof(_QINVCL), &_QINVCL);
		ret = clSetKernelArg(_kernels[PSEUDOINV_KER], 3, sizeof(_B), &_B);
		checkErr(ret, "PseudoInverse::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		length = 3 * 43 + BumpHeight*BumpWidth;
		length_out = divUp(length, (STRIDE * 2));
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\ATc.cl"), string("ATcKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[ATC_KER], 0, sizeof(_PseudoInverseCL), &_PseudoInverseCL);
		ret = clSetKernelArg(_kernels[ATC_KER], 1, sizeof(_B), &_B);
		ret = clSetKernelArg(_kernels[ATC_KER], 2, sizeof(_bufSolve1CL), &_bufSolve1CL);
		ret = clSetKernelArg(_kernels[ATC_KER], 3, sizeof(length), &length);
		ret = clSetKernelArg(_kernels[ATC_KER], 4, sizeof(length_out), &length_out);
		checkErr(ret, "ATcKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		length = divUp(3 * 43 + BumpHeight*BumpWidth, (STRIDE * 2));
		length_out = divUp(length, (STRIDE * 2));
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Reduce2.cl"), string("Reduce2Kernel"), _context, _device));
		ret = clSetKernelArg(_kernels[REDSOLVE_KER], 0, sizeof(_bufSolve1CL), &_bufSolve1CL);
		ret = clSetKernelArg(_kernels[REDSOLVE_KER], 1, sizeof(_bufSolve2CL), &_bufSolve2CL);
		ret = clSetKernelArg(_kernels[REDSOLVE_KER], 2, sizeof(length), &length);
		ret = clSetKernelArg(_kernels[REDSOLVE_KER], 3, sizeof(length_out), &length_out);
		checkErr(ret, "Reduce2Kernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		float sigma_d = 1.0;
		float sigma_r = 500.0;
		int size = 0;
		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Bilateral.cl"), string("BilateralKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 0, sizeof(_depthBuffCL), &_depthBuffCL);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 1, sizeof(_depthCL), &_depthCL);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 2, sizeof(sigma_d), &sigma_d);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 3, sizeof(sigma_r), &sigma_r);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 4, sizeof(size), &size);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 5, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[BILATERAL_KER], 6, sizeof(cDepthWidth), &cDepthWidth);
		checkErr(ret, "BilateralKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");


		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\MedianFilter.cl"), string("MedianFilterKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 0, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 1, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 2, sizeof(_RGBMapBumpCL), &_RGBMapBumpCL);
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 3, sizeof(_RGBMapBumpSwapCL), &_RGBMapBumpSwapCL);
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 4, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[MEDIANFILTER_KER], 5, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "MedianFilterKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\BumpVMP.cl"), string("VMPKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[VMP_KER], 0, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 1, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 2, sizeof(_RGBMapCL), &_RGBMapCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 3, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 4, sizeof(_BumpSwapCL), &_BumpSwapCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 5, sizeof(_RGBMapBumpCL), &_RGBMapBumpCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 6, sizeof(_RGBMapBumpSwapCL), &_RGBMapBumpSwapCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 7, sizeof(_VMapBumpCL), &_VMapBumpCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 8, sizeof(_NMapBumpCL), &_NMapBumpCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 9, sizeof(_NaturalParamCL), &_NaturalParamCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 10, sizeof(_PriorCL), &_PriorCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 11, sizeof(_ChildsCL), &_ChildsCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 12, sizeof(_ParentsCL), &_ParentsCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 13, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 14, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 15, sizeof(_LabelsMaskCL), &_LabelsMaskCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 16, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 17, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[VMP_KER], 18, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[VMP_KER], 19, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[VMP_KER], 20, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[VMP_KER], 21, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "VMPKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\Graph.cl"), string("GraphKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[GRAPH_KER], 0, sizeof(_BumpCL), &_BumpCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 1, sizeof(_RGBMapBumpCL), &_RGBMapBumpCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 2, sizeof(_VMapCL), &_VMapCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 3, sizeof(_NMapCL), &_NMapCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 4, sizeof(_RGBMapCL), &_RGBMapCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 5, sizeof(_ChildsCL), &_ChildsCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 6, sizeof(_ParentsCL), &_ParentsCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 7, sizeof(_BlendshapeCoeffCL), &_BlendshapeCoeffCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 8, sizeof(_verticesBSCL), &_verticesBSCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 9, sizeof(_LabelsMaskCL), &_LabelsMaskCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 10, sizeof(_PoseCL), &_PoseCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 11, sizeof(_intrinsicCL), &_intrinsicCL);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 12, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 13, sizeof(cDepthWidth), &cDepthWidth);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 14, sizeof(BumpHeight), &BumpHeight);
		ret = clSetKernelArg(_kernels[GRAPH_KER], 15, sizeof(BumpWidth), &BumpWidth);
		checkErr(ret, "GraphKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		_kernels.push_back(LoadKernel(string("..\\..\\OpenCL Kernels\\InitVMP.cl"), string("InitKernel"), _context, _device));
		ret = clSetKernelArg(_kernels[INITVMP_KER], 0, sizeof(_ParentsCL), &_ParentsCL);
		ret = clSetKernelArg(_kernels[INITVMP_KER], 1, sizeof(cDepthHeight), &cDepthHeight);
		ret = clSetKernelArg(_kernels[INITVMP_KER], 2, sizeof(cDepthWidth), &cDepthWidth);
		checkErr(ret, "InitKernel::setArg()");
		_queue.push_back(clCreateCommandQueue(_context, _device, 0, &ret));
		checkErr(ret, "CommandQueue::CommandQueue()");

		///////////////////////////////////////////////////////
	}

	HeadOffV2(float *Calib, char *path) : _draw(true), _idx(0), _path(path) {
		// set up face tracker
		_sdm = unique_ptr<SDM>(new SDM("..\\..\\models\\tracker_model49_131.bin"));
		_hpe = unique_ptr<HPE>(new HPE());
		_restart = true;
		_minScore = 0.1f;

		_LabelsMask = cv::imread(string("LabelsMask.png"), CV_LOAD_IMAGE_UNCHANGED);

		_vertices = (MyPoint **)malloc(cDepthHeight*cDepthWidth*sizeof(MyPoint *));
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				_vertices[i*cDepthWidth + j] = NULL;
			}
		}

		//_Bump = (float *)malloc(BumpHeight * BumpWidth*sizeof(float));
		//memset(_Bump, 0, BumpHeight * BumpWidth*sizeof(float));
		//_WeightMap = (float *)malloc(3 * BumpHeight * BumpWidth*sizeof(float));
		//memset(_WeightMap, 0, 3 * BumpHeight * BumpWidth*sizeof(float));
		_verticesBump = (Point3DGPU *)malloc(BumpHeight * BumpWidth*sizeof(Point3DGPU));
		//for (int i = 0; i < BumpHeight; i++) {
		//	for (int j = 0; j < BumpWidth; j++) {
				//_verticesBump[i*BumpWidth + j] = new Point3DGPU();
		//		//_Bump[i*BumpWidth + j] = 0.0;
		//	}
		//}

		_pRect.x = 0;
		_pRect.y = 0;
		_pRect.width = cDepthWidth;
		_pRect.height = cDepthHeight;

		for (int i = 0; i < 9; i++)
			_intrinsic[i] = Calib[i];

		for (int i = 0; i < 49; i++)
			_BlendshapeCoeff[i] = 0.0;
		_BlendshapeCoeff[0] = 1.0;

		_Rotation = Eigen::Matrix3f::Identity();
		_Translation = Eigen::Vector3f::Zero();
		_Rotation_inv = Eigen::Matrix3f::Identity();
		_Translation_inv = Eigen::Vector3f::Zero();
	}

	~HeadOffV2() {
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				if (_vertices[i*cDepthWidth + j] != NULL)
					std::free(_vertices[i*cDepthWidth + j]);
			}
		}
		std::free(_vertices);

		std::free(_verticesBump);
		std::free(_triangles);
		std::free(_verticesList);

		std::free(_outbuff);
		std::free(_outbuffJTJ);
		std::free(_outbuffGICP);
		std::free(_outbuffReduce);
		std::free(_outbuffResolved);

		std::free(TABLE_I);
		std::free(TABLE_J);
		std::free(_Qinv);

		clReleaseMemObject(_depthCL);
		clReleaseMemObject(_VMapCL);
		clReleaseMemObject(_NMapCL);
		clReleaseMemObject(_RGBMapCL);
		clReleaseMemObject(_SegmentedCL);
		clReleaseMemObject(_intrinsicCL);
		clReleaseMemObject(_PoseCL);
		clReleaseMemObject(_BlendshapeCoeffCL);
		clReleaseMemObject(_verticesBumpCL);
		clReleaseMemObject(_verticesBSCL);
		clReleaseMemObject(_TrianglesCL);
		clReleaseMemObject(_VerticesCL);
		clReleaseMemObject(_WeightMapCL);
		clReleaseMemObject(_LabelsMaskCL);
		clReleaseMemObject(_BumpCL); 
		clReleaseMemObject(_BumpSwapCL);
		clReleaseMemObject(_bufCL);
		clReleaseMemObject(_OutBuffCL);
		clReleaseMemObject(_OutBuffJTJCL);
		clReleaseMemObject(_bufJTJCL);

		clReleaseMemObject(_VMapBumpCL);
		clReleaseMemObject(_NMapBumpCL);
		clReleaseMemObject(_RGBMapBumpCL);
		clReleaseMemObject(_RGBMapBumpSwapCL);


		clReleaseMemObject(_B);
		clReleaseMemObject(_PseudoInverseCL);
		clReleaseMemObject(_QINVCL);
		clReleaseMemObject(_bufGICPCL);
		clReleaseMemObject(_OutBuffGICPCL);
		clReleaseMemObject(_bufGICPCL);
		clReleaseMemObject(_bufReduce1CL);
		clReleaseMemObject(_bufReduce2CL);
		clReleaseMemObject(_bufSolve1CL);
		clReleaseMemObject(_bufSolve2CL);

		clReleaseMemObject(_LandMarksBumpCL);
		clReleaseMemObject(_LandMarksCL);

		if (_Ske != NULL) {
			delete _Ske;
		}

		if (_KinectV2 != NULL) {
			delete _KinectV2;
		}
		
	}

	inline void SetParam(float *Calib, char *path, int KinectVersion = 0) {
		_path = new char[strlen(path) + 1]; 
		strcpy_s(_path, (strlen(path) + 1)*sizeof(char), path);
		_draw = true;

		for (int i = 0; i < 11; i++)
			_intrinsic[i] = Calib[i];
		_KinectVersion = KinectVersion;

		_Kinect1 = false;
		_Kinect2 = false;
		if (KinectVersion == 1) { // Run with Live data from Kinect Version 1.8
			_Kinect1 = true;
		}
		if (KinectVersion == 2) { // Run with Live data from Kinect Version 1.8
			_Kinect2 = true;
		}

		clSetKernelArg(_kernels[VMAP_KER], 4, sizeof(int), &KinectVersion);
	}

	inline void SetCoeff(int anim_indx){
		/*if (_idx < 5)
			return;*/
		_Rotation_inv = Eigen::Matrix3f::Identity();
		_Translation_inv = Eigen::Vector3f::Zero();

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				_Pose[4 * i + j] = _Rotation_inv(j, i);
			}
			_Pose[12 + i] = _Translation_inv(i);
		}

		for (int i = 0; i < NB_BS; i++)
			_BlendshapeCoeff[i] = 0.0f;
		_BlendshapeCoeff[anim_indx] = 1.0f;
	}

	inline float *getBump() { return (float *) _Bump.data; }
	inline Point3DGPU *getVerticesBump() { return _verticesBump; }

	inline bool Push() {
		if (_depth.size() == FRAME_BUFFER_SIZE) {
			return false;
		}

		_depth.push(_imgD);
		_color.push(_imgC);
		_segmented_color.push(_imgS);

		if (!_restart) {
			_ptsQ.push(_pts);
		}
		else {
			_ptsQ.push(cv::Mat());
		}

		return true;
	}

	//Start Kinect 1 sensor
	void StartKinect();

	//Start Kinect 2 sensor
	void StartKinect2();

	void Draw(bool color = true, bool bump = true);
	void DrawBlendedMesh(vector<MyMesh *> Blendshape);
	void DrawQuad(int color = 0, bool bump = true);

	void DrawRect(bool color = true);

	void DrawLandmark(int i);

	// Load next RGB-D image
	int Load();

	void Pop() {
		if (!_depth.empty()) {
			_color.pop();
			_segmented_color.pop();
			_ptsQ.pop();
			_depth.pop();
		}
	}

	int LoadToSave(int k);

	int SaveData(int k);

	bool Compute3DData();

	// Compute normals for each pixel of the depth image
	void ComputeNormalesDepth();

	// Detect or track facial landmarks using supervised descent method
	void DetectFeatures(cv::CascadeClassifier *face_cascade, bool draw = true);

	// Re-scale all blendshapes to match user landmarks
	bool Rescale(vector<MyMesh *> Blendshape);

	// Roughly align all blendshapes to the detected facial landmarks
	bool AlignToFace(vector<MyMesh *> Blendshape, bool inverted);

	void ElasticRegistrationFull(vector<MyMesh *> Blendshape);

	// Perform elastic registration to match facial features
	void ElasticRegistration(vector<MyMesh *> Blendshape);

	// Transfer expression deformations to all blendshapes
	void ComputeAffineTransfo(vector<MyMesh *> Blendshape);

	// Update Bump image from input RGB-D image
	void GenerateBump(vector<MyMesh *> Blendshape, int x, int y, int width, int height);
	void GenerateBumpGPU(vector<MyMesh *> Blendshape, int x, int y, int width, int height);

	void GenerateBumpMedianGPU(vector<MyMesh *> Blendshape, int x, int y, int width, int height);

	void GenerateBumpVMPGPU(vector<MyMesh *> Blendshape, int x, int y, int width, int height);

	void ComputeNormales(int x, int y, int width, int height);

	void InitLabels();

	void ComputeLabels(MyMesh *TheMesh);

	//Rigid registration of input RGB-D frame to 3D model
	void Register(vector<MyMesh *> Blendshape);
	void RegisterGPU(MyMesh *RefMesh);

	// Estimate animation blendshape coefficients
	void EstimateBlendShapeCoefficientsPR(vector<MyMesh *> Blendshape);
	// Estimate animation blendshape coefficients
	void EstimateBlendShapeCoefficientsPRGPU(vector<MyMesh *> Blendshape);

	// Estimate animation blendshape coefficients
	void EstimateBlendShapeCoefficientsGaussNewton(vector<MyMesh *> Blendshape);

	// Set landmark indices to fit initial RGB-D image
	void SetLandmarks(MyMesh *RefMesh);

	void LoadAnimatedModel();

	void AnimateModel();

	void LoadCoeffPose(char *path);

private:

	// 3D input data
	MyPoint **_vertices;
	cv::Mat _depthIn;
	cv::Mat _VMap;
	cv::Mat _NMap;
	cv::Mat _RGBMap;
	cv::Mat _segmentedIn;

	cv::Mat _imgD;
	cv::Mat _imgC;
	cv::Mat _imgS;
	cv::Mat _pts;

	cv::Mat _VMapBump;
	cv::Mat _NMapBump;
	cv::Mat _RGBMapBump;
	cv::Mat _WeightMap;

	cv::Mat _Bump; // deviation in milimeters
	cv::Mat _BumpSwap;


	////////////// OpenCL memory ////////////////////
	cl_mem _depthCL;
	cl_mem _depthBuffCL;
	cl_mem _VMapCL;
	cl_mem _NMapCL;
	cl_mem _RGBMapCL;
	cl_mem _SegmentedCL;
	cl_mem _intrinsicCL;
	cl_mem _PoseCL;
	cl_mem _QINVCL;
	cl_mem _B;
	cl_mem _PseudoInverseCL;

	cl_mem _BlendshapeCoeffCL;
	cl_mem _verticesBumpCL;
	cl_mem _VMapBumpCL;
	cl_mem _NMapBumpCL;
	cl_mem _RGBMapBumpCL;
	cl_mem _RGBMapBumpSwapCL;
	cl_mem _verticesBSCL;

	cl_mem _TrianglesCL;
	cl_mem _VerticesCL;
	cl_mem _WeightMapCL;
	cl_mem _LabelsMaskCL;

	cl_mem _BumpCL;
	cl_mem _BumpSwapCL;

	cl_mem _bufCL;
	cl_mem _OutBuffCL;
	cl_mem _bufJTJCL;
	cl_mem _OutBuffJTJCL;

	cl_mem _bufGICPCL;
	cl_mem _OutBuffGICPCL;
	cl_mem _bufReduce1CL;
	cl_mem _bufReduce2CL;
	cl_mem _bufSolve1CL;
	cl_mem _bufSolve2CL;

	cl_mem _LandMarksBumpCL;
	cl_mem _LandMarksCL;

	cl_mem _TableiCL;
	cl_mem _TablejCL;
	cl_mem _TableGICPiCL;
	cl_mem _TableGICPjCL;

	cl_mem _NbMatchesCL;

	cl_mem _NaturalParamCL;
	cl_mem _PriorCL;

	cl_mem _ParentsCL;
	cl_mem _ChildsCL;

	cl_event _evtVMap;
	////////////////////////////////////////////////


	// RGB-D data
	queue<cv::Mat> _depth;
	queue<cv::Mat> _color;
	queue<cv::Mat> _segmented_color;

	cv::Mat _LabelsMask;

	// Bump image data
	Point3DGPU *_verticesBump;
	BYTE *_RGB;
	//float *_WeightMap;
	float *_Vtx[49];

	double *_outbuff;
	double *_outbuffJTJ;
	double *_outbuffGICP;
	float *_outbuffReduce;
	float *_outbuffResolved;

	// Parameters for facial landmark detection/tracking
	unique_ptr<SDM> _sdm;
	unique_ptr<HPE> _hpe;

	cv::Mat _prevPts;
	cv::Mat _landmarks;
	cv::Mat _landmarksBump;
	bool _restart;
	float _minScore;
	queue<cv::Mat> _ptsQ;
	facio::HeadPose _hp;

	cv::Rect _pRect;

	vector<vector<Eigen::Matrix3f>> _TransfoExpression;
	vector<SpMat> _MatList1;
	vector<SpMat> _MatList2;

	Eigen::Vector3f _Translation;
	Eigen::Matrix3f _Rotation;

	Eigen::Vector3f _Translation_inv;
	Eigen::Matrix3f _Rotation_inv;

	vector<Eigen::Vector3f> _TranslationWindow;
	vector<Eigen::Matrix3f> _RotationWindow;
	vector<float> _BSCoeff[NB_BS];

	float _BlendshapeCoeff[49];

	float _intrinsic[11];
	float _Pose[16];
	float *_Qinv;
	int _nbMatches;

	// some flags
	bool _draw;
	int _idx;
	char *_path;

	int _lvl;
	int _max_iter[3];
	int _max_iterPR[3];
};


#endif