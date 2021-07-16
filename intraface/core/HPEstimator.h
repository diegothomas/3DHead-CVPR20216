#ifndef __HP_ESTIMATOR_H__
#define __HP_ESTIMATOR_H__


#include <intraface/Macros.h>
//#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>

using namespace std;

namespace facio {
	/** @ingroup core
	*/

	struct HeadPose {
		enum {
			PITCH=0, YAW, ROLL
		};
		cv::Mat rot; ///< head rotation matrix used for pre-multiplying
		cv::Point3f angles; ///< head rotation angles: pitch, yaw, roll
		cv::Point3f xyz; ///< head translation
	} ;

	/** @ingroup core
	*/
	/// <summary>
	/// An abstract class that provides a template for various head pose estimation algorithms
	/// </summary>
	class HPEstimator {

	public:

		/// <summary>
		/// Initializes a new instance of the <see cref="HPEstimator"/> class.
		/// </summary>
		HPEstimator() : 
			m_focalLength(-1.f),
			m_prinPoint(cv::Point2f(0.f,0.f))
		{}

		/// <summary>
		/// Initializes a new instance of the <see cref="HPEstimator"/> class.
		/// </summary>
		/// <param name="focalLength">The focal length of the camera in pixels.</param>
		/// <param name="prinPoint">The principle point of the camera.</param>
		/*HPEstimator(float focalLength, cv::Point2f prinPoint) :
			m_focalLength(focalLength),
			m_prinPoint(prinPoint)
		{}*/

		/// <summary>
		/// Finalizes an instance of the <see cref="HPEstimator"/> class.
		/// </summary>
		virtual ~HPEstimator() {}

		/// <summary>
		/// A pure virtual function that estimates the head pose
		/// </summary>
		/// <param name="X">The image projection of the facial landmarks.</param>
		/// <param name="pose">The head pose.</param>
		virtual void estimateHP(const cv::Mat& X, HeadPose& pose) = 0;

	protected:
		float m_focalLength;
		cv::Point2f m_prinPoint;

	};
}



#endif
