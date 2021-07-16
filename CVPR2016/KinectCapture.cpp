/**************************************************************************/
/* Author: Pavankumar Vasu Anasosalu									  */
/* Date:   March 08, 2013												  */
/* Last Modified: April 23, 2013										  */
/* Purpose: Contains the class definition for skeleton tracking			  */
/**************************************************************************/

/**** Includes ****/
#include "stdafx.h"
#include "KinectCapture.h"

/**** Constructor ****/
SkeletonTrack::SkeletonTrack()
{
	// initializing all the pointers and variables
	DWORD widthd = 0;
	DWORD heightd = 0;
	
	NuiImageResolutionToSize(depthResolution, widthd, heightd);
	dwidth  = static_cast<LONG>(widthd);
    dheight = static_cast<LONG>(heightd);
	
	NuiImageResolutionToSize(colorResolution, widthd, heightd);
	cwidth  = static_cast<LONG>(widthd);
    cheight = static_cast<LONG>(heightd);

    colordata        = (unsigned char *)  malloc(cwidth*cheight*4*sizeof(unsigned char));
	depthdata        = (unsigned short *) malloc(dwidth*dheight*sizeof(unsigned short));
	colorCoordinates = (LONG*)            malloc(dwidth*dheight*2*sizeof(LONG));

	CtoDdiv = cwidth/dwidth;
}

/**** Destructor ****/
SkeletonTrack::~SkeletonTrack()
{
	if (NULL != sensor)
    {
        sensor->NuiShutdown();
        sensor->Release();
    }

	free(colordata);
	free(depthdata);
	free(colorCoordinates);
}

/**** Initialization of Kinect Sensor ****/
HRESULT SkeletonTrack::initKinect()
{
	INuiSensor * pNuiSensor;
	HRESULT hr;

	int iSensorCount = 0;
	hr = NuiGetSensorCount(&iSensorCount);
	if (FAILED(hr))
	{
		return hr;
	}

	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i)
	{
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))
		{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)
		{
			sensor = pNuiSensor;
			break;
		}

		// This sensor wasn't OK, so release it since we're not using it
		pNuiSensor->Release();
	}

	if (NULL != sensor)
	{
		// Initialize the Kinect and specify that we'll be using depth
		hr = sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR);
		if (SUCCEEDED(hr))
		{
			// Create an event that will be signaled when depth data is available
			NextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Open a depth image stream to receive depth frames
			hr = sensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_DEPTH,
				depthResolution,
				NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE,
				2,
				NextDepthFrameEvent,
				&depthStream);

			// Create an event that will be signaled when color data is available
			NextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Initialize sensor to open up color stream
			hr = sensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_COLOR,
				colorResolution,
				0,
				2,
				NextColorFrameEvent,
				&rgbStream);

			if (FAILED(hr)) return hr;

			/*INuiColorCameraSettings *camSettings;
			hr = sensor->NuiGetColorCameraSettings(&camSettings);

			if (FAILED(hr))      return hr;

			hr = camSettings->SetAutoExposure(FALSE);
			if (FAILED(hr))      return hr;

			hr = camSettings->SetAutoWhiteBalance(FALSE);
			if (FAILED(hr))      return hr;

			hr = camSettings->SetWhiteBalance(4500);
			if (FAILED(hr))      return hr;*/
		}
		
	}

	if (NULL == sensor || FAILED(hr))
	{
		return hr;
	}

	return S_OK;
}

/**** Function to get the depth frame from the hardware ****/
HRESULT SkeletonTrack::getDepth(unsigned short *dest)
{
	NUI_IMAGE_FRAME imageFrame; 
	NUI_LOCKED_RECT LockedRect; 
	HRESULT hr;                 

	hr = sensor->NuiImageStreamGetNextFrame(depthStream,0,&imageFrame);
	if(FAILED(hr)) 		return hr;

	INuiFrameTexture *texture = imageFrame.pFrameTexture;
	hr = texture->LockRect(0,&LockedRect,NULL,0);
	if(FAILED(hr)) 		return hr;

	// Now copy the data to our own memory location
	if(LockedRect.Pitch != 0)
	{
		const GLushort* curr = (const unsigned short*) LockedRect.pBits;

		// copy the texture contents from current to destination
		memcpy( dest, curr, sizeof(unsigned short)*(dwidth*dheight) );
	}

	hr = texture->UnlockRect(0);
	if(FAILED(hr)) 		return hr;

	hr = sensor->NuiImageStreamReleaseFrame(depthStream, &imageFrame);
	if(FAILED(hr)) 		return hr;

	return S_OK;
}

/**** Function to get the color frame from the hardware ****/
HRESULT SkeletonTrack::getColor(unsigned char *dest)
{
	NUI_IMAGE_FRAME imageFrame; // structure containing all the metadata about the frame
	NUI_LOCKED_RECT LockedRect; // contains the pointer to the actual data
	HRESULT hr;                 // Error handling

	hr = sensor->NuiImageStreamGetNextFrame(rgbStream,0,&imageFrame);
	if(FAILED(hr))		return hr;

	INuiFrameTexture *texture = imageFrame.pFrameTexture;
	hr = texture->LockRect(0,&LockedRect,NULL,0);
	if(FAILED(hr))      return hr;

	// Now copy the data to our own memory location
	if(LockedRect.Pitch != 0)
	{
		const BYTE* curr = (const BYTE*) LockedRect.pBits;

		// copy the texture contents from current to destination
		memcpy( dest, curr, sizeof(BYTE)*(cwidth*cheight*4) );
	}

	hr = texture->UnlockRect(0);
	if(FAILED(hr))      return hr;

	hr = sensor->NuiImageStreamReleaseFrame(rgbStream, &imageFrame);
	if(FAILED(hr))      return hr;

	return S_OK;
}

/**** Function to map color frame to the depth frame ****/
HRESULT SkeletonTrack::MapColorToDepth(BYTE* colorFrame, USHORT* depthFrame)
{
	HRESULT hr;

	// Find the location in the color image corresponding to the depth image
	hr = sensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
		colorResolution,
		depthResolution,
		dwidth*dheight,
		depthFrame,
		(dwidth*dheight)*2,
		colorCoordinates);

	if(FAILED(hr))    return hr;

	return S_OK;
}

/**** Function to synchronize the frame capture ****/
void SkeletonTrack::getKinectData()
{
	bool needToMapColorToDepth = false;
	HRESULT hr;

	while(true) {
		if( WAIT_OBJECT_0 == WaitForSingleObject(NextDepthFrameEvent, 0) )
		{
			// if we have received any valid new depth data we proceed to obtain new color data
			if ( (hr = getDepth(depthdata)) == S_OK )
			{
				if( WAIT_OBJECT_0 == WaitForSingleObject(NextColorFrameEvent, 0) )
				{
					// if we have received any valid new color data we proceed to extract skeletal information
					if ( (hr = getColor(colordata)) == S_OK )
					{
						MapColorToDepth((BYTE*)colordata, (USHORT*)depthdata); 
						return;
					}
				}
			}
		} 	
	}
}

/**** Function of Utils class to perform cross product ****/
template<class TYPE>
void SkeletonTrackUtils::cross_product(TYPE *p1, TYPE *p2, TYPE *cross)
{
	// this function basically takes the cross product of two vectors
	cross[0] = (p1[1]*p2[2]) - (p1[2]*p2[1]);
	cross[1] = (p1[2]*p2[0]) - (p1[0]*p2[2]);
	cross[2] = (p1[0]*p2[1]) - (p1[1]*p2[0]);
}

/**** Function of Utils class to perform Matrix Multiplication ****/
template<class TYPE>
void SkeletonTrackUtils::MatrixMul(TYPE *in_vect, TYPE *out_vect, Matrix4 &T)
{
	TYPE i_vector[4], o_vector[4];
	
	i_vector[0] = in_vect[0];
	i_vector[1] = in_vect[1];
	i_vector[2] = in_vect[2];
	i_vector[3] = 1.0;

	o_vector[0] = T.M11*i_vector[0] + T.M12*i_vector[1] + T.M13*i_vector[2] + T.M14*i_vector[3];
	o_vector[1] = T.M21*i_vector[0] + T.M22*i_vector[1] + T.M23*i_vector[2] + T.M24*i_vector[3];
	o_vector[2] = T.M31*i_vector[0] + T.M32*i_vector[1] + T.M33*i_vector[2] + T.M34*i_vector[3];
	o_vector[3] = T.M41*i_vector[0] + T.M42*i_vector[1] + T.M43*i_vector[2] + T.M44*i_vector[3];

	out_vect[0] = o_vector[0] / o_vector[3];
	out_vect[1] = o_vector[1] / o_vector[3];
	out_vect[2] = o_vector[2] / o_vector[3];
}

/**** Function to normalize vector ****/
template<class TYPE>
void SkeletonTrackUtils::Normalize(TYPE *in_vect, TYPE *out_vect)
{
	double mag = sqrt(pow(in_vect[0],2) + pow(in_vect[1],2) + pow(in_vect[2],2));

	out_vect[0] = (TYPE)in_vect[0] / mag;
	out_vect[1] = (TYPE)in_vect[1] / mag;
	out_vect[2] = (TYPE)in_vect[2] / mag;
}
