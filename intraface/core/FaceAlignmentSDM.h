#ifndef __FACE_ALIGNMENT_SDM_H__
#define __FACE_ALIGNMENT_SDM_H__


#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <intraface/Macros.h>
#include <intraface/core/imgproc.h>
#include <intraface/core/XXDescriptor.h>
#include <intraface/core/binary_file_io.h>
#include <intraface/core/SimilarityWarper.h>
#include <intraface/core/ResizeWarper.h>
#include <intraface/core/WibuWrapper.h>

using namespace std;

namespace facio {

	enum ContourType {
		NO_CONTOUR,
		WITH_CONTOUR
	};

	/** @ingroup core
	*/
	/// <summary>
	/// Face alignment using Supervised Descent Method
	/// </summary>
	template<class Descriptor, ContourType CT = NO_CONTOUR>
	class DLLDIR FaceAlignmentSDM {

	public:
		/// <summary>
		/// Initializes a new instance of the <see cref="FaceAlignmentSDM"/> class.
		/// </summary>
		/// <param name="model">The model file.</param>
		/// <param name="offset">
		/// Optional parameter, the offset between your face detector output and OpenCV's. 
		/// Say the output of your face detector is (x,y,w,h). After applying the offset,
		/// it becomes (x+offset.x, y+offset.y, w*offset.width, h*offset.height).
		/// </param>
		/// <exception cref="std::runtime_error">The model file not found.</exception> 
		/// <exception cref="std::bad_alloc">Cannot construct Descriptor or Warper object.</exception> 
		FaceAlignmentSDM(const char* model, const cv::Rect_<float>& offset = cv::Rect_<float>(0.f, 0.f, 1.f, 1.f));

		/// <summary>
		/// Finalizes an instance of the <see cref="FaceAlignmentSDM"/> class.
		/// </summary>
		~FaceAlignmentSDM();

		/// <summary>
		/// Tracks facial landmarks.
		/// </summary>
		/// <param name="image">The image (must be in CV_8U).</param>
		/// <param name="prev">The previous landmarks (2xn).</param>
		/// <param name="landmarks">The predicted landmarks (2xn).</param>
		/// <param name="score">The confidence score of the prediction.</param>
		/// <exception cref="std::runtime_error">The input image is not in CV_8U.</exception> 
		void track(const cv::Mat& image, const cv::Mat& prev, cv::Mat& landmarks, float& score);

		/// <summary>
		/// Detects facial landmarks.
		/// </summary>
		/// <param name="image">The image must be in CV_8U).</param>
		/// <param name="face">
		/// The face square (x,y,w,h). (x,y) is the upper left corner of face region. 
		/// (w,h) are the width and height of the face region.
		/// </param>
		/// <param name="landmarks">The predicted landmarks (2xn).</param>
		/// <param name="score">The confidence score of the prediction.</param>
		/// <exception cref="std::runtime_error">The input image is not in CV_8U.</exception> 
		void detect(const cv::Mat& image, const cv::Rect& face, cv::Mat& landmarks, float& score);


	private:
		const static int CLASS_ID = Core + 100;
		inline void load(const char *model);
		void load_(const char *model);
		class FaceAlignmentSDM_;
		FaceAlignmentSDM_ *m_fa;

	};
}

#endif
