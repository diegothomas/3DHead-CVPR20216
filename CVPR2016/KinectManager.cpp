#include "stdafx.h"
#include "KinectManager.h"

/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT KinectV2Manager::InitializeDefaultSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get the depth reader
		IDepthFrameSource* pDepthFrameSource = NULL;
		// Initialize the Kinect and get the color reader
		IColorFrameSource* pColorFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		}

		SafeRelease(pDepthFrameSource);

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&CoordinateMapper);
		}

		SafeRelease(pColorFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!" << endl;
		return E_FAIL;
	}

	return hr;
}

/// <summary>
/// Handle new depth data
/// <param name="nTime">timestamp of frame</param>
/// <param name="pBuffer">pointer to frame data</param>
/// <param name="nWidth">width (in pixels) of input image data</param>
/// <param name="nHeight">height (in pixels) of input image data</param>
/// <param name="nMinDepth">minimum reliable depth</param>
/// <param name="nMaxDepth">maximum reliable depth</param>
/// </summary>
void KinectV2Manager::ProcessDepth(INT64 nTime, const UINT16* pBuffer, int nWidth, int nHeight, USHORT nMinDepth, USHORT nMaxDepth)
{
	// Make sure we've received valid data
	if (m_pDepthRGBX && pBuffer && (nWidth == cDepthWidth) && (nHeight == cDepthHeight))
	{
		RGBQUAD* pRGBX = m_pDepthRGBX;

		// end pixel is start + width*height - 1
		const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

		while (pBuffer < pBufferEnd)
		{
			USHORT depth = *pBuffer;

			// To convert to a byte, we're discarding the most-significant
			// rather than least-significant bits.
			// We're preserving detail, although the intensity will "wrap."
			// Values outside the reliable depth range are mapped to 0 (black).

			// Note: Using conditionals in this loop could degrade performance.
			// Consider using a lookup table instead when writing production code.
			BYTE intensity = static_cast<BYTE>((depth >= nMinDepth) && (depth <= nMaxDepth) ? (depth % 256) : 0);

			pRGBX->rgbRed = intensity;
			pRGBX->rgbGreen = intensity;
			pRGBX->rgbBlue = intensity;

			++pRGBX;
			++pBuffer;
		}
	}
}

bool KinectV2Manager::UpdateManager() {
	if (!m_pColorFrameReader || !m_pDepthFrameReader)
	{
		return false;
	}

	//HRESULT hr = m_pKinectSensor->get_CoordinateMapper(&CoordinateMapper);

	IColorFrame* pColorFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	HRESULT hr = E_FAIL;
	HRESULT hrc = E_FAIL;

	while (!SUCCEEDED(hr)) {
		hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);
		if (!SUCCEEDED(hr))
			SafeRelease(pDepthFrame);
	}

	float nTimeD = clock();
	float nTimeC = 0;
	while (!SUCCEEDED(hrc)) {
		hrc = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);
		if (!SUCCEEDED(hrc)) {
			SafeRelease(pColorFrame);
		}
	}

	if (SUCCEEDED(hr) && SUCCEEDED(hrc))
	{
		INT64 nTimeD = 0;
		INT64 nTimeC = 0;
		IFrameDescription* pDepthDescription = NULL;
		IFrameDescription* pColorDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxReliableDistance = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		RGBQUAD *pBufferC = NULL;


		hr = pDepthFrame->get_RelativeTime(&nTimeD);
		//cout << "time depth: " << nTime << endl;
		hr = pColorFrame->get_RelativeTime(&nTimeC);
		//cout << "time color: " << nTime << endl;

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_FrameDescription(&pDepthDescription);
			hrc = pColorFrame->get_FrameDescription(&pColorDescription);
		}

		if (SUCCEEDED(hr) && SUCCEEDED(hrc))
		{
			hr = pDepthDescription->get_Width(&nWidth);
			//FrameIn->AssertDepthWidth(nWidth);
			hrc = pColorDescription->get_Width(&nWidth);
			//FrameIn->AssertColorWidth(nWidth);
		}

		if (SUCCEEDED(hr) && SUCCEEDED(hrc))
		{
			hr = pDepthDescription->get_Height(&nHeight);
			//FrameIn->AssertDepthHeight(nHeight);
			hrc = pColorDescription->get_Height(&nHeight);
			//FrameIn->AssertColorHeight(nHeight);
		}

		if (SUCCEEDED(hr) && SUCCEEDED(hrc))
		{
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
			//FrameIn->setDepthMinReliableDistance(nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr) && SUCCEEDED(hrc))
		{
			hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
			//FrameIn->setDepthMaxReliableDistance(nDepthMaxReliableDistance);
			hrc = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr) && SUCCEEDED(hrc))
		{
			hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
			memcpy(depthdata, pBuffer, cDepthHeight * cDepthWidth * sizeof(UINT16));
			//FrameIn->setDepthBuffer(pBuffer, nTimeD + dec);

			if (imageFormat == ColorImageFormat_Bgra)
			{
				hrc = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pBufferC));
				memcpy(colordata, pBufferC, cColorHeight * cColorWidth * sizeof(RGBQUAD));
				//FrameIn->setColorBuffer(pBufferC, nTimeC);
			}
			else
			{
				//pBufferC = FrameIn->getColorBuffer();
				nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hrc = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(colordata), ColorImageFormat_Bgra);
			}
		}

		SafeRelease(pDepthDescription);
		SafeRelease(pColorDescription);
	}

	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);

	if (SUCCEEDED(hr) && SUCCEEDED(hrc))
	{
		hr = CoordinateMapper->MapDepthFrameToCameraSpace(cDepthHeight * cDepthWidth, depthdata, cDepthHeight * cDepthWidth, depth2CameraSpacePoints);
		hr = CoordinateMapper->MapDepthFrameToColorSpace(cDepthHeight * cDepthWidth, depthdata, cDepthHeight * cDepthWidth, colorCoordinates);
	}

	return SUCCEEDED(hr) && SUCCEEDED(hrc);
}