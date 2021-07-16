#ifndef __MACROS_H__
#define __MACROS_H__

#include "FacioConfig.h"

/*! \def DLLDIR_EX
\brief A macro indicating whether to export or import dll information
*/
//#define DLLDIR_EX

/*! \def _MSC_VER
\brief Microsoft Visual Studio version number
*/
#ifdef _MSC_VER 
#  ifdef DLLDIR_EX

#    define DLLDIR  __declspec(dllexport)   // export dll information
#  else
#    define DLLDIR  __declspec(dllimport)   // import dll information
#  endif 
#else
#  define DLLDIR
#endif

//#define DLLDIR


namespace facio {

	enum Module {
		Core = 0000,
		Attr = 1000,
		Gaze = 2000,
		Emo  = 3000,
		Recg = 4000
	};
	const int SINGLE_ROW = 0; /// opencv reduce function
	const int SINGLE_COLUMN = 1; /// opencv reduce function
	const int L_EYE_L_CORNER = 19;
	const int L_EYE_R_CORNER = 22;
	const int R_EYE_L_CORNER = 25;
	const int R_EYE_R_CORNER = 28;

	const int L_EYE_START = 19; // inclusive
	const int L_EYE_END   = 25; // exclusive
	const int R_EYE_START = 25;
	const int R_EYE_END   = 31;

	const int L_EYEBROW_START = 0;
	const int L_EYEBROW_END   = 5;
	const int R_EYEBROW_START = 5;
	const int R_EYEBROW_END   = 10;

	const int NOSE_START = 10;
	const int NOSE_END   = 19;

	const int MOUTH_START = 31;
	const int MOUTH_END   = 49;

	const int JAW_START = 49;
	const int JAW_END   = 66;

	const int CURRENT_VERSION=INTRAFACE_VERSION; ///< version number.
	const float EPS = 0.000001f; ///< epsilon, a small value to avoid division by zero.
	const float PI = 3.1415926f; ///< \f$\pi\f$
	const float HALF_PI = 1.5707963f; ///< \f$\frac{\pi}{2}\f$
	const float INVERSE_PI = 1.f/PI; ///< \f$\frac{1}{\pi}\f$
	const float INVERSE_2PI = 0.5f/PI; ///< \f$\frac{1}{2\pi}\f$
	const float RAD_TO_DEG_FACTOR = 57.2957795f; ///< radian to degree conversion factor
	const float DEG_TO_RAD_FACTOR = 0.01745329f; ///< degree to radian conversion factor

	inline int fast_floor(float x) {
		int i = (int)x;
		if (i > x) --i;
		return i;
	}

	inline float fast_absf(float x) {
		return (x >= 0.f) ? x : -x;
	}
}


#endif
