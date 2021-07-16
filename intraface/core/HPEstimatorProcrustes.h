#ifndef __HP_ESTIMATOR_PROCRUSTES_H__
#define __HP_ESTIMATOR_PROCRUSTES_H__


#include <intraface/Macros.h>
#include <memory>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <intraface/core/HPEstimator.h>
#include <stdexcept>

using namespace std;

namespace facio {

	
	class DLLDIR HPEstimatorProcrustes: public HPEstimator {

	public:
		/// <summary>
		/// Initializes a new instance of the <see cref="HPEstimatorProcrustes"/> class.
		/// </summary>
		HPEstimatorProcrustes();

		~HPEstimatorProcrustes();

		/// <summary>
		/// Initializes a new instance of the <see cref="HPEstimatorProcrustes"/> class.
		/// </summary>
		/// <param name="f">The focal length of the camera.</param>
		/// <param name="pp">The prinple point of the camera.</param>
		//HPEstimatorProcrustes(float f, cv::Point2f pp);

		/// <summary>
		/// Estimates the head pose using procrustes algorithm.
		/// </summary>
		/// <param name="p2D">The 2D landmarks (2x49).</param>
		/// <param name="pose">
		/// The computed head pose. only rotation angles are computed. 
		///	No translation computed.
		/// </param>
		/// <returns>status</returns>
		void estimateHP(const cv::Mat& p2D, HeadPose& pose);

	private:
		const static int CLASS_ID = Core + 200;
		class HPEstimatorProcrustes_;
		unique_ptr<HPEstimatorProcrustes_> m_hpe;
	};

}



#endif
