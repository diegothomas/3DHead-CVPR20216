#ifndef __XX_DESCRIPTOR_H__
#define __XX_DESCRIPTOR_H__

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <utility>
#include <memory>
#include <Eigen/Core>
#include <intraface/Macros.h>
#include <intraface/core/WibuWrapper.h>

using namespace std;

namespace facio {

	template<int NOB,int NSB,int WINSIZE>
	class DLLDIR XXDescriptor {

	public: 
		/// <summary>
		/// Initializes a new instance of XXDescriptor class.
		/// </summary>
		XXDescriptor();

		~XXDescriptor();

		/// <summary>
		/// Computes image descriptors for the input image centered at each landmark location.
		/// </summary>
		/// <param name="image">The image (double grayscale). Each element must be within [0,255].</param>
		/// <param name="landmark">The landmark(nx2).</param>
		/// <param name="output">The output (float).</param>
		/// <param name="winsize">The patch size.</param>
		void compute(const cv::Mat& image, const Eigen::MatrixXf& landmark, Eigen::VectorXf& output) const;


	private:
		const static int CLASS_ID=Core+0;
		class XXDescriptor_;
		unique_ptr<XXDescriptor_> m_xxd;
		inline void init();
		void init_();
	};

	typedef XXDescriptor<8,4,32> AlignmentFeature;
	typedef XXDescriptor<8,3,24> IrisFeature;
}


#endif
