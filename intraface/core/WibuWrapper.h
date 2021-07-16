#ifndef __WIBU_WRAPPER__
#define __WIBU_WRAPPER__

#include <stdexcept>
#include <intraface/Macros.h>
#ifdef USE_IXP
	#include <wibu/wibuixap.h>
#endif
#include <intraface/FacioConfig.h>

namespace facio {
	class DLLDIR WibuWrapper {
	public:
#ifdef USE_IXP
		inline static void prefix(int id) {
			if (WupiDecryptCode(id) != 1)
			{      
				throw std::runtime_error("License Error");
			}
		}
	
		inline static void suffix(int id) {
			WupiEncryptCode(id);     
		}
#else
		inline static void prefix(int id) {}
	
		inline static void suffix(int id) {}

#endif
	};

}


#endif
