// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "HeadOffV2.h"
#include "Mesh.h"

GLuint window;

/*** Camera variables for OpenGL ***/
GLfloat intrinsics[16] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float Calib[11] = { 580.8857f, 583.317f, 319.5f, 239.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 8000.0f }; // Kinect data
float Znear = 0.05f;
float Zfar = 10.0f;
GLfloat light_pos[] = { 0.0, 0.0, 2.0, 0.0 }; //{ 1.0, 1.0, 0.0, 0.0 };
GLfloat ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat diffuseLight[] = { 0.8f, 0.8f, 0.8f, 1.0f };

// angle of rotation for the camera direction
float anglex = 0.0f;
float angley = 0.0f;

// actual vector representing the camera's direction
float lx = 0.0f, ly = 0.0f, lz = -1.0f;
float lxStrap = -1.0f, lyStrap = 0.0f, lzStrap = 0.0f;

// XZ position of the camera
float x = 0.0f, y = 0.0f, z = 0.1f; //0.15f;//
float deltaAnglex = 0.0f;
float deltaAngley = 0.0f;
float deltaMove = 0;
float deltaStrap = 0;
int xOrigin = -1;
int yOrigin = -1;

bool Running = false;
bool stop = false;
clock_t current_time;
clock_t last_time;
float my_count;
float fps;
bool first;
int anim_indx = 0;
bool save_img = false;
bool inverted = false;

HeadOffV2 *TheHeadOff = NULL;
cv::CascadeClassifier face_cascade;

mutex imageMutex;
mutex bumpMutex;
condition_variable condv;

MyMesh *TheMesh;
std::vector<MyMesh *> Blendshape;

thread t1;
thread t2;
thread t3;
thread t4;
thread t5;
thread t6;
int idim = 0;
bool color = true;
bool bump = true;
bool quad = true;
bool ready_to_bump = false;

bool terminated;
bool isTerminated() { return terminated; }

void KinectLive();

typedef CL_API_ENTRY cl_int(CL_API_CALL *P1)(const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

CL_API_ENTRY cl_int(CL_API_CALL *myclGetGLContextInfoKHR) (const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) = NULL;


int Init() {
	cl_device_id device_id = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	//cl::Context context = NULL;
	cl_context context;
	cl_context_properties platform, hRC, hDC;

	/*** Initialise OpenCL and OpenGL interoperability***/

	// Get platforms.
	cl_uint lNbPlatformId = 0;
	clGetPlatformIDs(0, 0, &lNbPlatformId);

	if (lNbPlatformId == 0)
	{
		std::cerr << "Unable to find an OpenCL platform." << std::endl;
		return -1;
	}


	// Loop on all platforms.
	std::vector< cl_platform_id > platformList(lNbPlatformId);
	clGetPlatformIDs(lNbPlatformId, platformList.data(), 0);

	std::cerr << "Platform number is: " << lNbPlatformId << std::endl;
	char platformVendor[10240];
	clGetPlatformInfo(platformList[0], (cl_platform_info)CL_PLATFORM_VENDOR, 10240, platformVendor, NULL);
	std::cerr << "Platform is by: " << platformVendor << "\n";

	myclGetGLContextInfoKHR = (P1)clGetExtensionFunctionAddressForPlatform(platformList[0], "clGetGLContextInfoKHR");

	cl_uint lNbDeviceId = 0;
	clGetDeviceIDs(platformList[0], CL_DEVICE_TYPE_GPU, 0, 0, &lNbDeviceId);

	if (lNbDeviceId == 0)
	{
		return -1;
	}

	std::vector< cl_device_id > lDeviceIds(lNbDeviceId);
	clGetDeviceIDs(platformList[0], CL_DEVICE_TYPE_GPU, lNbDeviceId, lDeviceIds.data(), 0);


	//Additional attributes to OpenCL context creation
	//which associate an OpenGL context with the OpenCL context 
	cl_context_properties props[] =
	{
		//OpenCL platform
		CL_CONTEXT_PLATFORM, (cl_context_properties)platformList[0],
		//OpenGL context
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		//HDC used to create the OpenGL context
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		0
	};

	cl_int lError = CL_SUCCESS;
	context = clCreateContext(props, 1, &lDeviceIds[0], 0, 0, &lError);
	if (lError != CL_SUCCESS)
	{
		cout << "device unsupported" << endl;
	}


	// Load the cascade
	string faceDetectionModel("..\\..\\models\\haarcascade_frontalface_alt2.xml");
	if (!face_cascade.load(faceDetectionModel))
		throw runtime_error("Error loading face detection model.");

	TheHeadOff = new HeadOffV2(context, lDeviceIds[0]);


////////////////////////////////// ALGO FOR SIMPLIFICATION ////////////////////////////////////
////////////////////////////////// ALGO FOR SIMPLIFICATION ////////////////////////////////////
////////////////////////////////// ALGO FOR SIMPLIFICATION ////////////////////////////////////


	//TheMesh = new MyMesh(&TheHeadOff->_verticesList[0], &TheHeadOff->_triangles[0]);
	//TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\Neutralm.obj"), false);
	//Blendshape.push_back(TheMesh);

	////TheMesh = new MyMesh(&TheHeadOff->_verticesList[4325], &TheHeadOff->_triangles[1]);
	////TheMesh->LoadS(string("NeutralmS.obj"));
	////TheMesh->Write(string("Neutralm.obj"));
	////Blendshape.push_back(TheMesh);

	////Blendshape[1]->Map(Blendshape[0]);

	//TheMesh = new MyMesh(&TheHeadOff->_verticesList[7366], &TheHeadOff->_triangles[1]);
	//TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\MyTemplate\\Neutralm.obj"), true);
	//Blendshape.push_back(TheMesh);

	//Blendshape[0]->Map(Blendshape[1]);
	//exit(0);

	//char filename[100];
	//int indx_vtx = 2;
	//for (int i = 0; i < 10; i++) {
	//	TheMesh = new MyMesh(&TheHeadOff->_verticesList[indx_vtx * 4325], &TheHeadOff->_triangles[0]);
	//	//TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\0") + to_string(i) + string("m.obj"));
	//	TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\MyTemplate\\0") + to_string(i) + string("m.obj"));
	//	Blendshape[0]->Modify(TheMesh);
	//	TheMesh->Write(string("0") + to_string(i) + string("m.obj"));
	//}

	//for (int i = 10; i < 48; i++) {
	//	TheMesh = new MyMesh(&TheHeadOff->_verticesList[indx_vtx * 4325], &TheHeadOff->_triangles[0]);
	//	//TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\") + to_string(i) + string("m.obj"));
	//	TheMesh->Load(string("C:\\Diego\\Projects\\Data\\blendshapes\\MyTemplate\\") + to_string(i) + string("m.obj"));
	//	Blendshape[0]->Modify(TheMesh);
	//	TheMesh->Write(to_string(i) + string("m.obj"));
	//}

	//exit(0);


////////////////////////////////// END ALGO FOR SIMPLIFICATION ////////////////////////////////////
////////////////////////////////// END ALGO FOR SIMPLIFICATION ////////////////////////////////////
////////////////////////////////// END ALGO FOR SIMPLIFICATION ////////////////////////////////////



	TheMesh = new MyMesh(&TheHeadOff->_verticesList[0], &TheHeadOff->_triangles[0]);
	TheMesh->Load(string("..\\..\\blendshapes\\MyTemplate\\Neutralm.obj"), true);
	Blendshape.push_back(TheMesh);
	// remove 02,03,04,05,06,07,08,09,10,11,12,13,17,18,19,22,26,27,34,35,42
	char filename[100];
	int indx_vtx = 1;
	for (int i = 0; i < 2; i++) {
		TheMesh = new MyMesh(&TheHeadOff->_verticesList[indx_vtx * 4325], &TheHeadOff->_triangles[0]);
		TheMesh->Load(string("..\\..\\blendshapes\\MyTemplate\\0") + to_string(i) + string("m.obj"));
		Blendshape.push_back(TheMesh);
		indx_vtx++;
	}

	for (int i = 14; i < 48; i++) {
		if (i == 17 || i == 18 || i == 19 || i == 22 || i == 26 || i == 27 || i == 34 || i == 35 || i == 42)
			continue;
		TheMesh = new MyMesh(&TheHeadOff->_verticesList[indx_vtx * 4325], &TheHeadOff->_triangles[0]);
		TheMesh->Load(string("..\\..\\blendshapes\\MyTemplate\\") + to_string(i) + string("m.obj"));
		Blendshape.push_back(TheMesh);
		indx_vtx++;
	}
	cout << Blendshape.size() << endl;

	//cout << std::thread::hardware_concurrency() << endl;

	TheHeadOff->ComputeAffineTransfo(Blendshape); 
	TheHeadOff->InitLabels();
	//TheHeadOff->ComputeLabels(Blendshape[0]);

	current_time = clock();
	last_time = clock();
	my_count = 0.0;
	first = true;

	/*** Initialize OpenGL ***/
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glClearColor(1.0f, 1.0f, 1.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations
	// enable color tracking
	glEnable(GL_COLOR_MATERIAL);
	// set material properties which will be assigned by glColor
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	intrinsics[0] = 2.0 * Calib[0] / cDepthWidth;
	intrinsics[5] = 2.0 * Calib[1] / cDepthHeight;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0;
	intrinsics[14] = -2.0*(Zfar * Znear) / (Zfar - Znear);

	return 0;
}

void reshape(int width_in, int height_in)
{
	glViewport(0, 0, width_in, height_in);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
	// Set up camera intrinsics
	glLoadMatrixf(intrinsics);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void computePos(float deltaMove, float deltaStrap) {

	x += deltaMove * lx * 0.1f + deltaStrap * lxStrap * 0.1f;
	y += deltaMove * ly * 0.1f + deltaStrap * lyStrap * 0.1f;
	z += deltaMove * lz * 0.1f + deltaStrap * lzStrap * 0.1f;
}

void saveimg(int x, int y, int id, char *path) {
	float *image = new float[3 * cDepthWidth*cDepthHeight];
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadBuffer(GL_FRONT);
	glReadPixels(x, y, cDepthWidth, cDepthHeight, GL_RGB, GL_FLOAT, image);

	cv::Mat imagetest(cDepthHeight, cDepthWidth, CV_8UC3);
	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[2] = unsigned char(255.0*image[3 * (i*cDepthWidth + j)]);
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[1] = unsigned char(255.0*image[3 * (i*cDepthWidth + j) + 1]);
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[0] = unsigned char(255.0*image[3 * (i*cDepthWidth + j) + 2]);
		}
	}

	char filename[100];
	sprintf_s(filename, "%s%d.png", path, id);
	cv::imwrite(filename, imagetest);

	delete[] image;
	image = 0;
}

void produce()
{
	while (true) {

		int res = TheHeadOff->Load();

		if (res == 3) { // end of sequence
			delete TheHeadOff;
			exit(1);
		}

		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;

		//if (!lck.owns_lock())
		//	lck.lock();

		//TheHeadOff->Push();

		//lck.unlock();

		/*my_count++;
		current_time = clock();
		if ((current_time - last_time) / CLOCKS_PER_SEC > 1.0) {
			fps = my_count / ((current_time - last_time) / CLOCKS_PER_SEC);
			last_time = current_time;
			my_count = 0.0;
			cout << "fps: " << fps << endl;

		}*/


		//continue;
	//	if (res == 0){
	//		break;
	//	}
		if (res == 2) {
				TheHeadOff->DetectFeatures(&face_cascade);

				while (!isTerminated()) {
					if (!lck.owns_lock())
						lck.lock();

					if (TheHeadOff->Push()) {
						lck.unlock();
						break;
					}

					lck.unlock();
				}

		}
	}
}

void consume() {
	chrono::time_point<std::chrono::system_clock> start_t, end_t;
	while (true) {

		/*if (TheHeadOff->SaveData() < 0)
			continue;*/

		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;

		if (!lck.owns_lock())
			lck.lock();

		//TheHeadOff->Pop();
		//lck.unlock();

		//continue;*/

		//current_time = clock();
		if (!TheHeadOff->Compute3DData()) {
			//TheHeadOff->SetCoeff(anim_indx);
			//TheHeadOff->AnimateModel();
			//current_time = clock();
			//last_time = current_time;
			//while ((current_time - last_time) / CLOCKS_PER_SEC < 0.1)
			//	current_time = clock();
			//save_img = true;
			lck.unlock();
			continue;
		}
		TheHeadOff->Pop();
		//cout << "time: " << ((clock() - current_time) / CLOCKS_PER_SEC) << endl;
		lck.unlock();

		//continue;

		if (first) {
			// Re-scale all blendshapes to match user
			if (TheHeadOff->Rescale(Blendshape)) {
				if (TheHeadOff->AlignToFace(Blendshape, inverted)) {
					TheHeadOff->Register(Blendshape);
					////TheHeadOff->ElasticRegistration(Blendshape);
					TheHeadOff->ElasticRegistrationFull(Blendshape);
					////TheHeadOff->ElasticRegistration(Blendshape);
					//TheHeadOff->GenerateBumpGPU(Blendshape, 0, 0, BumpWidth, BumpHeight);
					cout << "OK" << endl;
					TheHeadOff->GenerateBumpVMPGPU(Blendshape, 0, 0, BumpWidth, BumpHeight);
					TheHeadOff->SetLandmarks(Blendshape[0]);
					first = false;
				}
			}
		}
		else {
			//return;
			//current_time = clock();
			TheHeadOff->RegisterGPU(Blendshape[0]);
			//cout << "RegisterGPU timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;
			//TheHeadOff->EstimateBlendShapeCoefficientsGaussNewton(Blendshape);
			//TheHeadOff->EstimateBlendShapeCoefficientsPR(Blendshape);
			//current_time = clock();
			TheHeadOff->EstimateBlendShapeCoefficientsPRGPU(Blendshape);
			//TheHeadOff->AnimateModel();
			//cout << "EstimateBlendShapeCoefficientsPRGPU timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;
			//current_time = clock();
			//TheHeadOff->GenerateBumpGPU(Blendshape, 0, 0, BumpWidth, BumpHeight);
			TheHeadOff->GenerateBumpVMPGPU(Blendshape, 0, 0, BumpWidth, BumpHeight);
			//cout << "GenerateBumpGPU timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;
			//cout << "consume timing: " << float((clock() - current_time)) / CLOCKS_PER_SEC << endl;
		}

		/*current_time = clock();
		last_time = current_time;
		while ((current_time - last_time) / CLOCKS_PER_SEC < 0.1)
			current_time = clock();*/
		if (save_data)
			save_img = true;
		/*current_time = clock();
		last_time = current_time;
		while ((current_time - last_time) / CLOCKS_PER_SEC < 0.1)
			current_time = clock();*/

		/*int kkk = 0;
		while (save_img) {
			kkk++;
		}*/
		my_count++;
		current_time = clock();
		if ((current_time - last_time) / CLOCKS_PER_SEC > 1.0) {
			fps = my_count / ((current_time - last_time) / CLOCKS_PER_SEC);
			last_time = current_time;
			my_count = 0.0;
			cout << "fps: " << fps << endl;

		}
	}
}


void stream()
{
	while (true) {

		int res = TheHeadOff->Load();

		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;


		while (true) {
			if (!lck.owns_lock())
				lck.lock();

			if (TheHeadOff->Push() || terminated) {
				lck.unlock();
				break;
			}

			lck.unlock();
		}

		my_count++;
		current_time = clock();
		if ((current_time - last_time) / CLOCKS_PER_SEC > 1.0) {
		fps = my_count / ((current_time - last_time) / CLOCKS_PER_SEC);
		last_time = current_time;
		my_count = 0.0;
		cout << "fps: " << fps << endl;

		}

	}
}
void SaveInput(int k) {
	chrono::time_point<std::chrono::system_clock> start_t, end_t;
	while (true) {
		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;

		bool oksave = false;
		if (!lck.owns_lock())
			lck.lock();

		if (TheHeadOff->LoadToSave(k) == 1) 
			oksave = true;

		TheHeadOff->Pop();
		lck.unlock();

		if (oksave)
			TheHeadOff->SaveData(k);

	}
}

void consumePlayBack() {
	//chrono::time_point<std::chrono::system_clock> start_t, end_t;
	//while (true) {
	//	unique_lock<mutex> lck(imageMutex);
	//	if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;
	//	if (!lck.owns_lock())
	//		lck.lock();

		char filename_buff[100];
		sprintf_s(filename_buff, "%s\\Animation", dest_name);
		//TheHeadOff->LoadCoeffPose(filename_buff);
		TheHeadOff->SetCoeff(anim_indx);
		TheHeadOff->AnimateModel();

		/*current_time = clock();
		last_time = current_time;
		while ((current_time - last_time) / CLOCKS_PER_SEC < 0.1)
			current_time = clock();*/
		save_img = true;

	//	lck.unlock();
	//}
}

void bumpmapping(int x, int y, int width, int height) {
	chrono::time_point<std::chrono::system_clock> start_t, end_t;
	while (true) {
		unique_lock<mutex> lck(bumpMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;
		if (ready_to_bump) {
			TheHeadOff->GenerateBump(Blendshape, x, y, width, height);
			ready_to_bump = false;
		}
	}
}

void displayOFF(void) {

	//produce();
	//consume();
	//consumePlayBack();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
	// Set up camera intrinsics
	glLoadMatrixf(intrinsics);

	glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Set the camera
	gluLookAt(x, y, z,
		x + lx, y + ly, z + lz,
		0.0f, 1.0f, 0.0f);

	//unique_lock<mutex> lck(imageMutex);
	//if (!lck.owns_lock())
	//	lck.lock();
	//glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	//glPointSize(6.0);
	//for (int i = 0; i < 43; i++)
	//	TheHeadOff->DrawLandmark(i);
	///*TheHeadOff->DrawLandmark(19);
	//TheHeadOff->DrawLandmark(22);
	//TheHeadOff->DrawLandmark(24);
	//TheHeadOff->DrawLandmark(28);
	//TheHeadOff->DrawLandmark(31);
	//TheHeadOff->DrawLandmark(37);
	//TheHeadOff->DrawLandmark(10);
	//TheHeadOff->DrawLandmark(16);
	//TheHeadOff->DrawLandmark(13);*/
	//glPointSize(1.0);

	//glViewport(0, 0, cDepthWidth, cDepthHeight);
	//TheHeadOff->DrawRect(false);

	glEnable(GL_LIGHTING);
	GLfloat ambientLightq[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat diffuseLightq[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLightq);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLightq);
	glEnable(GL_LIGHT0);

	/*glViewport(0, cDepthHeight, cDepthWidth, cDepthHeight);
	TheHeadOff->DrawRect();

	glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	TheHeadOff->DrawRect(false);*/

	//GLfloat light_pos2[] = { -1.0f, 1.0f, 0.0f, 0.0f };
	//glLightfv(GL_LIGHT1, GL_POSITION, light_pos2);
	////glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLightq);
	//glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLightq);
	//glEnable(GL_LIGHT1);

	//GLfloat light_pos3[] = { 0.0f, -1.0f, 1.0f, 0.0f };
	//glLightfv(GL_LIGHT2, GL_POSITION, light_pos3);
	////glLightfv(GL_LIGHT2, GL_AMBIENT, ambientLightq);
	//glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuseLightq);
	//glEnable(GL_LIGHT2);


	//glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	// Draw neutral blendshape over input
	//Blendshape[0]->Draw();

	// Set the camera
	//gluLookAt(x, y, z - 0.5,
	//	x + lx, y + ly, z + lz,
	//	0.0f, 1.0f, 0.0f);

	glViewport(0, 0, cDepthWidth, cDepthHeight); // Normal image
	if (quad)
		TheHeadOff->DrawQuad(0, bump);
	else
		TheHeadOff->Draw(color, bump);

	//glViewport(2 * cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	//TheHeadOff->DrawBlendedMesh(Blendshape);


	//Blendshape[anim_indx]->Draw(TheHeadOff->getBump());
	//glPointSize(6.0);
	//for (int i = 0; i < 43; i++)
	//	Blendshape[anim_indx]->DrawLandmark(i);

	//glViewport(cDepthWidth, 0, cDepthWidth, cDepthHeight); // RGB image
	//if (quad)
	//	TheHeadOff->DrawQuad(1, bump);
	//else
	//	TheHeadOff->Draw(color, bump);

	//glViewport(2 * cDepthWidth, 0, cDepthWidth, cDepthHeight);  // Geo image
	//TheHeadOff->DrawQuad(2, bump);


	//glViewport(0, 0, cDepthWidth, cDepthHeight);
	//TheHeadOff->Draw(false, false);
	
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);

	//glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	//glPointSize(6.0);
	///*for (int i = 0; i < 43; i++)
	//	Blendshape[0]->DrawLandmark(i);*/
	//Blendshape[0]->DrawLandmark(19);
	//Blendshape[0]->DrawLandmark(22);
	//Blendshape[0]->DrawLandmark(25);
	//Blendshape[0]->DrawLandmark(28);
	//Blendshape[0]->DrawLandmark(31);
	//Blendshape[0]->DrawLandmark(37);
	//Blendshape[0]->DrawLandmark(10);
	//Blendshape[0]->DrawLandmark(16);
	//Blendshape[0]->DrawLandmark(13);
	//glPointSize(1.0);

	if (save_img) {
		char filename_buff[100];
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\Normal", dest_name);
		saveimg(0, 0, idim, filename_buff);
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\RGB", dest_name);
		saveimg(cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\Geo", dest_name);
		saveimg(2 * cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\Original", dest_name);
		saveimg(2 * cDepthWidth, cDepthHeight, idim, filename_buff);

		/*sprintf_s(filename_buff, "%s\\Retargeted\\Normal", dest_name);
		saveimg(0, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Retargeted\\RGB", dest_name);
		saveimg(cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Retargeted\\Geo", dest_name);
		saveimg(2 * cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Input\\RGB", dest_name);
		saveimg(0, cDepthHeight, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Input\\Normal", dest_name);
		saveimg(cDepthWidth, cDepthHeight, idim, filename_buff);*/
		idim++;
		save_img = false;
		anim_indx = (anim_indx + 1) % 28;
	}
	//lck.unlock();
	
	glutSwapBuffers();
	glutPostRedisplay();
	return;
}

void displayLIVE(void) {
	glutSwapBuffers();
	glutPostRedisplay();
	return;
}

void display(void) {

	if (!Running) {
		glutSwapBuffers();
		glutPostRedisplay();
		return;
	}

	if (deltaMove || deltaStrap)
		computePos(deltaMove, deltaStrap);

	if (TheHeadOff) {
		displayOFF();
	}
	else {
		displayLIVE();
	}
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 'c':
		color = !color;
		break;
	case 'b':
		bump = !bump;
		break;
	case 'q':
		quad = !quad;
		break;
	case 's':
		KinectLive();
		break;
	case 27 /* Esc */:
		terminated = true;
		if (Running) {
			try {
				t1.join();
				t2.join();
				//t3.join();
			}
			catch (exception& e) {
				cerr << e.what() << endl;
			}
		}
		delete TheHeadOff;
		exit(1);
	}
	// 
}

void keyUp(unsigned char key, int x, int y) {
	switch (key) {
	case 'a':
		break;
	default:
		break;
	}
}

void pressKey(int key, int xx, int yy) {

	switch (key) {
	case GLUT_KEY_UP: deltaMove = 0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_DOWN: deltaMove = -0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_LEFT: deltaStrap = 0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_RIGHT: deltaStrap = -0.5f / 3.0f/*(fps/30.0)*/; break;
	}
}

void releaseKey(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_LEFT:
	case GLUT_KEY_RIGHT:
	case GLUT_KEY_UP:
	case GLUT_KEY_DOWN: deltaMove = 0; deltaStrap = 0; break;
	}
}

void mouseMove(int x, int y) {

	// this will only be true when the left button is down
	if (xOrigin >= 0 || yOrigin >= 0) {

		// update deltaAngle
		deltaAnglex = (x - xOrigin) * 0.001f;
		deltaAngley = (y - yOrigin) * 0.001f;

		// update camera's direction
		lx = sin(anglex + deltaAnglex);
		ly = cos(anglex + deltaAnglex) * sin(-(angley + deltaAngley));
		lz = -cos(anglex + deltaAnglex) * cos(-(angley + deltaAngley));

		// update camera's direction
		lxStrap = -cos(anglex + deltaAnglex);
		lyStrap = sin(anglex + deltaAnglex) * sin(-(angley + deltaAngley));
		lzStrap = -sin(anglex + deltaAnglex) * cos(-(angley + deltaAngley));
	}
}

void mouseButton(int button, int state, int x, int y) {

	// only start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {
		// when the button is released
		if (state == GLUT_UP) {
			anglex += deltaAnglex;
			angley += deltaAngley;
			xOrigin = -1;
			yOrigin = -1;
		}
		else  {// state = GLUT_DOWN
			xOrigin = x;
			yOrigin = y;
		}
	}
}

/***** Function to handle right click of Mouse for subwindow 1*****/
void KinectLive()
{
	Running = true;
	/*** Live Kinect data ***/
	intrinsics[0] = 2.0*580.8857 / 640.0; intrinsics[5] = 2.0*583.317 / 480.0;
	intrinsics[8] = 0.0;
	intrinsics[9] = 0.0;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0; intrinsics[14] = -2.0*(Zfar*Znear) / (Zfar - Znear);
	cout << "Running on live kinect." << endl;

	//cDepthWidth = 640;
	//cDepthHeight = 480;

	char path[] = "";
	TheHeadOff->SetParam(Calib, path, 1);
	TheHeadOff->StartKinect();

	inverted = true;
	t1 = thread(produce);
	t2 = thread(consume);
	//t1 = thread(stream);
	//t2 = thread(SaveInput, 0);
	//t3 = thread(SaveInput, 1);
}

void KinectOffLine()
{
	/*** Offline Kinect data ***/
	Running = true;
	intrinsics[0] = 2.0*580.8857 / 640.0; intrinsics[5] = 2.0*583.317 / 480.0;
	intrinsics[8] = 0.0;
	intrinsics[9] = 0.0;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0; intrinsics[14] = -2.0*(Zfar*Znear) / (Zfar - Znear);
	cout << "Running on live kinect." << endl;

	//cDepthWidth = 640;
	//cDepthHeight = 480;

	char path[] = "Seq\\KinectV1";
	TheHeadOff->SetParam(Calib, path, 3);

	inverted = true;
	t1 = thread(produce);
	t2 = thread(consume);
}

void KinectV2Live()
{
	/*** Live Kinect V2 data ***/
	Calib[0] = 320.0f; Calib[1] = 320.0f; Calib[2] = 250.0f; Calib[3] = 217.0f; Calib[4] = 0.0f; Calib[5] = 0.0f; Calib[6] = 0.0f; Calib[7] = 0.0f; Calib[8] = 0.0f; Calib[9] = 1.0f; Calib[10] = 10000.0f;
	intrinsics[0] = 2.0f * Calib[0] / cDepthWidth;
	intrinsics[5] = 2.0f * Calib[1] / cDepthHeight;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0f;
	intrinsics[14] = -2.0f*(Zfar * Znear) / (Zfar - Znear);

	/* Check for Kinect */
	//inverted = true; 
	char path[] = "";
	TheHeadOff->SetParam(Calib, path, 2);
	TheHeadOff->StartKinect2();
	t1 = thread(produce);
	t2 = thread(consume);
	//t1 = thread(stream);
	//t2 = thread(SaveInput, 0);
	//t3 = thread(SaveInput, 1);
}

void KinectV2OffLine()
{
	/*** Offline Kinect data ***/
	// for "C:\\Diego\\Projects\\Data\\KinectV2\\SetFace" and KV2-2
	//Calib[0] = 357.324f; Calib[1] = 362.123f; Calib[2] = 250.123f; Calib[3] = 217.526f; Calib[4] = 0.0f; Calib[5] = 0.0f; Calib[6] = 0.0f; Calib[7] = 0.0f; Calib[8] = 0.0f; Calib[9] = 4.5f; Calib[10] = 255.0f*255.0f;

	// For Seq
	//Calib[0] = 320.0f; Calib[1] = 320.0f; Calib[2] = 250.0f; Calib[3] = 217.0f; Calib[4] = 0.0f; Calib[5] = 0.0f; Calib[6] = 0.0f; Calib[7] = 0.0f; Calib[8] = 0.0f; Calib[9] = 1.0f; Calib[10] = 10000.0f; 
	//KV2-2
	Calib[0] = 357.324f; Calib[1] = 362.123f; Calib[2] = 250.123f; Calib[3] = 217.526f; Calib[4] = 0.0f; Calib[5] = 0.0f; Calib[6] = 0.0f; Calib[7] = 0.0f; Calib[8] = 0.0f; Calib[9] = 1.0f; Calib[10] = 10000.0f;
	
	intrinsics[0] = 2.0f * Calib[0] / cDepthWidth;
	intrinsics[5] = 2.0f * Calib[1] / cDepthHeight;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0f;
	intrinsics[14] = -2.0f*(Zfar * Znear) / (Zfar - Znear);
	cout << "Running on SetFace." << endl;

	char path[] = "Seq\\KinectV2-2";
	TheHeadOff->SetParam(Calib, path, 3); //Seq\\Kinect2-a "C:\\Diego\\Projects\\Data\\KinectV2\\SetFace"

	Running = true;
	//inverted = true;
	t1 = thread(produce);
	t2 = thread(consume);
}

void PlayBack() {
	cout << "Playing back " << dest_name << endl;

	Blendshape.clear();
	TheMesh = new MyMesh(&TheHeadOff->_verticesList[0], &TheHeadOff->_triangles[0]);
	TheMesh->Load(string(dest_name) + string("\\DeformedMeshes\\Neutral.obj"));
	Blendshape.push_back(TheMesh);
	char filename[100];
	int indx_vtx = 1;
	for (int i = 0; i < 27; i++) {
		TheMesh = new MyMesh(&TheHeadOff->_verticesList[indx_vtx * 4325], &TheHeadOff->_triangles[0]);
		TheMesh->Load(string(dest_name) + string("\\DeformedMeshes\\") + to_string(i) + string(".obj"));
		Blendshape.push_back(TheMesh);
		indx_vtx++;
	}

	TheHeadOff->LoadAnimatedModel();
	TheHeadOff->AnimateModel();
	Running = true;
	//t1 = thread(consumePlayBack);
}

void Right_menu(int val)
{
	switch (val)
	{
	case 0:
		KinectLive();
		break;
	case 1:
		KinectOffLine();
		break;
	case 2:
		KinectV2Live();
		break;
	case 3:
		KinectV2OffLine();
		break;
	case 4:
		PlayBack();
		break;
	default:
		break;
	}
	//Running = true;
}

int _tmain(int argc, _TCHAR* argv[])
{
	char *inGlut = new char[strlen("Bouh")];
	glutInit(&argc, &inGlut);

	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

	glutInitWindowSize(cDepthWidth, cDepthHeight); // (3 * cDepthWidth, 2 * cDepthHeight);

	window = glutCreateWindow("HeadModeler");

	if (Init() != 0)
		return 1;

	int menu_general = glutCreateMenu(Right_menu);
	glutAddMenuEntry("Kinect V1 live", 0);
	glutAddMenuEntry("Kinect V1 off-line", 1);
	glutAddMenuEntry("Kinect V2 live", 2);
	glutAddMenuEntry("Kinect V2 off-line", 3);
	glutAddMenuEntry("Play back", 4);

	glutAttachMenu(GLUT_RIGHT_BUTTON);

	glutReshapeFunc(reshape);
	glutDisplayFunc(display);

	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyUp);

	glutSpecialFunc(pressKey);
	glutSpecialUpFunc(releaseKey);

	// here are the two new functions
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

	glutMainLoop();

	return 0;
}

