#ifndef __SIMILARITY_WARPER_H__
#define __SIMILARITY_WARPER_H__


#include <intraface/Macros.h>
#include <intraface/core/procrustes.h>
#include <intraface/core/AffineWarper.h>
#include <opencv2/core/core.hpp>

using namespace std;

namespace facio {
	/** @ingroup core
	*/
	/// <summary>
	/// Image warping under similarity transformation.
	/// <remarks>
	/// Similarity transformation has 4 degree of freedom: translation (2), scale (1), and rotation (1).
	/// </remarks>
	/// </summary>
	class DLLDIR SimilarityWarper : public AffineWarper {

	public:
		
		/// <summary>
		/// Initializes a new instance of the <see cref="ResizeWarper"/> class.
		/// </summary>
		/// <param name="ref">The reference shape.</param>
		/// <exception cref="std::invalid_argument">The input matrix is empty.</exception> 
		SimilarityWarper(const cv::Mat& ref) : AffineWarper(ref) {}

		/// <summary>
		/// Finalizes an instance of the <see cref="SimilarityWarper"/> class.
		/// </summary>
		~SimilarityWarper() {}


	private:
		void computeWarpParameter(const cv::Mat& X);
		
	};
}

#endif
