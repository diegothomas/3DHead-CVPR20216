#ifndef __PROCRUSTES_H__
#define __PROCRUSTES_H__

#include <Eigen/Dense>
#include <intraface/Macros.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

using namespace std;

namespace facio {
	/** \addtogroup core
	*  @{
	*/

	/*! \typedef struct {
		cv::Mat R;
		cv::Mat T;
		cv::Mat S;
	} RigidMotion;

    \brief A type definition for rigid transformation.
	*/
	typedef struct {
		cv::Mat R; ///< rotation matrix 
		cv::Mat T; ///< translation vector
		cv::Mat S; ///< scaling matrix
	} RigidMotion;
	
	
	/*! 
	\fn void iso_procrustes(const cv::Mat& X, const cv::Mat& Y, RigidMotion& A)
    \brief Performs Procrustes between two point sets. 

		X and Y are related by: X = A.R*A.S*A.Y + A.T
    \param X The reference points (2xn).
    \param Y The query points (2xn).
	\param A Rigid motion computed that transfers Y to X.
	*/
	DLLDIR void iso_procrustes(const cv::Mat& X, const cv::Mat& Y, RigidMotion& A);

	
	/*! 
	\fn void noniso_procrustes(const cv::Mat& X, const cv::Mat& Y, RigidMotion& A)
    \brief Performs Procrustes between two point sets. 

		X and Y are related by: X = A.R*A.S*A.Y + A.T
	This function computes scales in both x and y axes while \ref iso_procrustes() 
	computes only one scale.
    \param X The reference points (2xn).
    \param Y The query points (2xn).
	\param A Rigid motion computed that transfers Y to X.
	*/
	DLLDIR void noniso_procrustes(const cv::Mat& X, const cv::Mat& Y, RigidMotion& A);


	inline void bw_transfer(const RigidMotion& A, cv::Mat& X) {
		X.row(0) -= A.T.at<float>(0);
		X.row(1) -= A.T.at<float>(1);
		cv::Mat S = (cv::Mat_<float>(2,2) << 1.f/A.S.at<float>(0), 0.f, 0.f, 1.f/A.S.at<float>(3));
		X = (A.R.t()*S)*X;
	}

	inline void fw_transfer(const RigidMotion& A, cv::Mat& X) {
		X = (A.R*A.S)*X;
		X.row(0) += A.T.at<float>(0);
		X.row(1) += A.T.at<float>(1);
		
	}

	/** @}*/
}

#endif
