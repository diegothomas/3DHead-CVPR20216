#ifndef __AFFINE_WARPER_H__
#define __AFFINE_WARPER_H__


#include <intraface/Macros.h>
#include <intraface/core/Warper.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <exception>

using namespace std;

namespace facio {
	/** @ingroup core
	*/
	/// <summary>
	/// Image warping under affine transformation.
	/// <remarks>
	/// Affine transformation has 6 degrees of freedom, often represented by a 2x3 matrix.
	/// </remarks>
	/// </summary>
	class DLLDIR AffineWarper : public Warper {

	public:

		/// <summary>
		/// Initializes a new instance of the <see cref="AffineWarper"/> class.
		/// </summary>
		/// <param name="ref">The reference shape.</param>
		/// <exception cref="std::invalid_argument">The input matrix is empty.</exception> 
		AffineWarper(const cv::Mat& ref) : Warper(ref) {}

		/// <summary>
		/// Finalizes an instance of the <see cref="AffineWarper"/> class.
		/// </summary>
		virtual ~AffineWarper() {}

		/// <summary>
		/// Warps the input image using affine transformation. 
		/// </summary>
		/// <remarks> 
		/// The affine matrix (2x3) is computed tby minimizing the L2 distance 
		/// between the reference shape and the input shape.
		/// </remarks>
		/// <param name="im">The image.</param>
		/// <param name="landmarks">The shape, represented by a set of landmarks (2xn).</param>
		/// <exception cref="std::invalid_argument">The input shape has different dimension with the reference shape.</exception> 
		virtual void warp(cv::Mat& im, cv::Mat& landmarks);

		/// <summary>
		/// Warps the shape back into the original image coordinate.
		/// </summary>
		/// <param name="landmarks">The warped shape, represented by a set of landmarks (2xn).</param>
		void warpBack(cv::Mat& landmarks) const;


	protected:
		cv::Mat m_A;

		virtual void computeWarpParameter(const cv::Mat& X);

	};
}

#endif
