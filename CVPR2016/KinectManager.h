#ifndef _KINMAN_H
#define _KINMAN_H

#pragma once

class KinectV2Manager
{
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;

public:
	/// Constructor
	KinectV2Manager() : m_pKinectSensor(NULL), m_pDepthFrameReader(NULL), m_pDepthRGBX(NULL) {
		// create heap storage for depth pixel data in RGBX format
		m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];

		colordata = (RGBQUAD *)malloc(cColorHeight * cColorWidth * sizeof(RGBQUAD));
		depthdata = (UINT16 *)malloc(cDepthWidth*cDepthHeight*sizeof(UINT16));
		colorCoordinates = (ColorSpacePoint *)malloc(cDepthHeight * cDepthWidth * sizeof(ColorSpacePoint));
		depth2CameraSpacePoints = (CameraSpacePoint *)malloc(cDepthHeight * cDepthWidth* sizeof(CameraSpacePoint));
	};

	/// Destructor
	~KinectV2Manager() {
		if (m_pDepthRGBX)
		{
			delete[] m_pDepthRGBX;
			m_pDepthRGBX = NULL;
		}

		free(colordata);
		free(depthdata);
		free(colorCoordinates);
		free(depth2CameraSpacePoints);

		// done with depth frame reader
		SafeRelease(m_pDepthFrameReader);

		// close the Kinect Sensor
		if (m_pKinectSensor)
		{
			m_pKinectSensor->Close();
		}

		SafeRelease(m_pKinectSensor);
	};

	/// Handles window messages, passes most to the class instance to handle
	static LRESULT CALLBACK MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	/// Handle windows messages for a class instance
	LRESULT CALLBACK        DlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	/// Initializes the default Kinect sensor
	//S_OK on success, otherwise failure code
	HRESULT                 InitializeDefaultSensor();

	bool UpdateManager();

	bool inline Available() {
		return m_pDepthFrameReader != NULL;
	}

	inline ICoordinateMapper* getMapper() { return CoordinateMapper; }

	RGBQUAD*    getColorframe()
	{
		return colordata;
	}

	UINT16*  getDepthframe()
	{
		return depthdata;
	}

	CameraSpacePoint*    getCameraCoord()
	{
		return depth2CameraSpacePoints;
	}

	ColorSpacePoint*    getColorCoord()
	{
		return colorCoordinates;
	}

private:

	// Current Kinect
	IKinectSensor*          m_pKinectSensor;

	// Depth reader
	IDepthFrameReader*      m_pDepthFrameReader;
	// Color reader
	IColorFrameReader*      m_pColorFrameReader;


	ICoordinateMapper *CoordinateMapper;

	// Direct2D
	//ImageRenderer*          m_pDrawDepth;
	//ID2D1Factory*           m_pD2DFactory;
	RGBQUAD*                m_pDepthRGBX;

	// Color and depth frames
	RGBQUAD*				colordata;
	UINT16*					depthdata;
	ColorSpacePoint *		colorCoordinates;
	CameraSpacePoint *		depth2CameraSpacePoints;

	/// Main processing function
	void                    Update();


	/// Handle new depth data
	void                    ProcessDepth(INT64 nTime, const UINT16* pBuffer, int nHeight, int nWidth, USHORT nMinDepth, USHORT nMaxDepth);

	/// Set the status bar message
	bool                    SetStatusMessage(_In_z_ WCHAR* szMessage, DWORD nShowTimeMsec, bool bForce);

	/// Get the name of the file where screenshot will be stored.
	HRESULT                 GetScreenshotFileName(_Out_writes_z_(nFilePathSize) LPWSTR lpszFilePath, UINT nFilePathSize);

	/// Save passed in image data to disk as a bitmap
	HRESULT                 SaveBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath);
};

#endif