#ifndef __RESIZE_WARPER_H__
#define __RESIZE_WARPER_H__


#include <intraface/Macros.h>
#include <intraface/core/procrustes.h>
#include <intraface/core/AffineWarper.h>
#include <opencv2/core/core.hpp>

using namespace std;

namespace facio {
	/** @ingroup core
	*/
	/// <summary>
	/// Image warping with scale changes only.
	/// <remarks>
	/// Resize transformation has 3 degree of freedom: translation (2), scale (1).
	/// </remarks>
	/// </summary>
	class DLLDIR ResizeWarper : public AffineWarper {

	public:
		/// <summary>
		/// Initializes a new instance of the <see cref="ResizeWarper"/> class.
		/// </summary>
		/// <param name="ref">The reference shape.</param>
		/// <exception cref="std::invalid_argument">The input matrix is empty.</exception> 
		ResizeWarper(const cv::Mat& ref);

		/// <summary>
		/// Finalizes an instance of the <see cref="ResizeWarper"/> class.
		/// </summary>
		~ResizeWarper() {}

		/// <summary>
		/// Scale the input image to match with the reference shape. 
		/// </summary>
		/// <param name="im">The image.</param>
		/// <param name="landmarks">The shape, represented by a set of landmarks (2xn).</param>
		/// <exception cref="std::invalid_argument">The input shape has different dimension with the reference shape.</exception> 
		void warp(cv::Mat& im, cv::Mat& landmarks);

	private:
		float m_scale;
		void computeWarpParameter(const cv::Mat& X);

	};
}

#endif
