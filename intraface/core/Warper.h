#ifndef __WARPER_H__
#define __WARPER_H__


#include <intraface/Macros.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

using namespace std;

namespace facio {
	/** @ingroup core
	*/
	/// <summary>
	/// An abstract class that implements different image warping methods
	/// </summary>
	class DLLDIR Warper {

	public:

		/// <summary>
		/// Initializes a new instance of the <see cref="Warper"/> class.
		/// </summary>
		/// <param name="ref">The reference shape.</param>
		/// <exception cref="std::invalid_argument">The input matrix is empty.</exception> 
		Warper(const cv::Mat& ref) {
			m_ref = ref.clone();
			if (ref.total() == 0)
				throw invalid_argument("SimilarityWarper::SimilarityWarper() empty matrix.");

			int n = ref.cols;
			m_ones = cv::Mat::ones(1,n,CV_32F);
		}

		/// <summary>
		/// Finalizes an instance of the <see cref="Warper"/> class.
		/// </summary>
		virtual ~Warper() {}

		/// <summary>
		/// A pure virtual function that warps the input image according to the reference shape
		/// </summary>
		/// <param name="im">The image.</param>
		/// <param name="landmarks">The shape (represented by a set of landmarks).</param>
		/// <exception cref="std::invalid_argument">The input shape has different dimension with the reference shape.</exception> 
		virtual void warp(cv::Mat& im, cv::Mat& landmarks) = 0;

		/// <summary>
		/// A pure virtual function that warps the shape back into the original image coordinate.
		/// </summary>
		/// <param name="landmarks">The warped shape (represented by a set of landmarks).</param>
		virtual void warpBack(cv::Mat& landmarks) const = 0;


	protected:
		void check(const cv::Mat& X) const  
		{
			if (X.cols != m_ref.cols)
				throw std::invalid_argument("Warper::check(): dimension of inputs do not match.");
		}

		cv::Mat m_ref;
		cv::Mat m_ones;

	};
}

#endif
