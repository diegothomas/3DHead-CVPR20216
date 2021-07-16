/**************************************************************************/
/* Author: Diego Thomas													  */
/* Date:   July 22, 2015												  */
/* Last Modified: July 22, 2015										  */
/* Purpose: Contains the class definition for kinect capture		  */
/**************************************************************************/

#ifndef __KINECTCAPTURE_H
#define __KINECTCAPTURE_H

#include "KinectManager.h"

//typedef struct
//{
//	float3 X_axis;
//	float3 Y_axis;
//	float3 Z_axis;
//}CoordAxis;

class SkeletonTrack
{
	/****** Class variable declarations*******/
private:
	// Resolution of the streams
	static const NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_640x480;
	static const NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_640x480;
	
	// Mapped color coordinates from color frame to depth frame
	LONG*                         colorCoordinates;
	
	// Event handlers
	HANDLE						  rgbStream;
	HANDLE						  depthStream;
	HANDLE						  NextDepthFrameEvent;
	HANDLE						  NextColorFrameEvent;

	// Variables related to resolution assigned in constructor
	int							  CtoDdiv;
	long int					  cwidth;
	long int					  dwidth;
	long int					  cheight;
	long int					  dheight;

	// Actual sensor connected to the Computer
	INuiSensor*				      sensor;
	
	// Color and depth frames
	unsigned char*				  colordata;
	unsigned short*				  depthdata;
		
	/****** Class function declarations *******/
public:

	/* Constructor */
	SkeletonTrack();

	/* Destructor */
	~SkeletonTrack();

	/* Function that identifies the First Kinect sensor connected to the PC */ 
	HRESULT initKinect();

	/* Function to get the depth frame from the hardware */ 
	HRESULT getDepth(unsigned short *dest);

	/* Function to get the color frame from the hardware */
	HRESULT getColor(unsigned char *dest);

	/* Function to map color frame to the depth frame */
	HRESULT MapColorToDepth(BYTE* colorFrame, USHORT* depthFrame); 

	/* Function to synchronize the frame capture */
	void getKinectData();

	/**** Getters for the private members of the class ****/
	
	long int getCwidth()
	{
		return cwidth;
	}

	long int getDwidth()
	{
		return dwidth;
	}

	long int getCheight()
	{
		return cheight;
	}

	long int getDheight()
	{
		return dheight;
	}

	int      getCtoDdiv()
	{
		return CtoDdiv;
	}

	BYTE*    getColorframe()
	{
		return colordata;
	}

	USHORT*  getDepthframe()
	{
		return depthdata;
	}

	LONG*    getColorCoord()
	{
		return colorCoordinates;
	}

};



/**** class for utilities/functionalites required by SkeletonTrack class ****/
class SkeletonTrackUtils
{
public:
	/* Function to perform the cross product of two vectors */
	template<class TYPE> 
	void cross_product(TYPE *p1, TYPE *p2, TYPE *cross);

	/* Function of Utils class to perform Matrix Multiplication */
	template<class TYPE>
    void MatrixMul(TYPE *in_vect, TYPE *out_vect, Matrix4 &T);

	/* Function to normalize vector */
	template<class TYPE>
	void Normalize(TYPE *in_vect, TYPE *out_vect);
};

#endif


