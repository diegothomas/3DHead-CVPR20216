#ifndef __IMG_PROC_H__
#define __IMG_PROC_H__

#include <algorithm>
#include <intraface/Macros.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>



namespace facio {
	
	inline void rgb2gray(const cv::Mat& image, cv::Mat& grayImage)
	{
		int c = image.channels();
		if (c==3)
#ifdef OPENCV_VERSION_GREATER_THAN300
			cvtColor(image,grayImage,cv::COLOR_BGR2GRAY);
#else
			cvtColor(image, grayImage, CV_BGR2GRAY);
#endif
		else if (c==4)
#ifdef OPENCV_VERSION_GREATER_THAN300
			cvtColor(image,grayImage,cv::COLOR_BGRA2GRAY);
#else
			cvtColor(image, grayImage, CV_BGRA2GRAY);
#endif
		else
			grayImage = image;
	}

	DLLDIR void imcrop(const cv::Mat& inputIm, cv::Mat& outputIm, const cv::Rect& inputROI, cv::Point& offset);

	template<typename T>
	inline void uncrop(const cv::Point& offset, cv::Mat& pts)
	{
		pts.row(0) += static_cast<T>(offset.x);
		pts.row(1) += static_cast<T>(offset.y);
	}

	// T is the type of pts
	template<typename T>
	void imcrop_around_pts(const cv::Mat& im, cv::Mat& pts, cv::Mat& cim, 
		cv::Point& offset,float borderX=0.f, float borderY=0.f)
	{
		cv::Mat topLeft, bottomRight, dim;
		cv::reduce(pts,topLeft,1,CV_REDUCE_MIN);
		cv::reduce(pts,bottomRight,1,CV_REDUCE_MAX);
		dim = bottomRight - topLeft + 1;

		cv::Rect roi;
		roi.x = fast_floor(topLeft.at<T>(0)-borderX*dim.at<T>(0));
		roi.y = fast_floor(topLeft.at<T>(1)-borderY*dim.at<T>(1));
		roi.width  = static_cast<int>(dim.at<T>(0)*(borderX+borderX+1));
		roi.height = static_cast<int>(dim.at<T>(1)*(borderY+borderY+1));

		imcrop(im, cim, roi, offset);
		T roix = static_cast<T>(offset.x);
		T roiy = static_cast<T>(offset.y);
		pts.row(0) -= roix;
		pts.row(1) -= roiy;	
	}
}



#endif