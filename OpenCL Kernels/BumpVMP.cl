#pragma OPENCL EXTENSION cl_amd_printf : enable

#define	MESSAGE_LENGTH 40
#define	NB_BS 28

__kernel void VMPKernel(__read_only image2d_t VMap, __read_only image2d_t NMap, __read_only image2d_t RGBMap,
						__read_only image2d_t Bump, __write_only image2d_t BumpSwap, 
						__read_only image2d_t RGBMapBump, __write_only image2d_t RGBMapBumpSwap,
						__write_only image2d_t VMapBump, __read_only image2d_t NMapBump,
						__global float *NaturalParam, 
						__global float *Prior, 
						__global int *Childs, 
						__global int *Parents, 
						__constant float *BlendshapeCoeff, 
						__global float *VerticesBS, 
						__read_only image2d_t LabelsMask, 
						__constant float *Pose, __constant float *calib, 
						int n, int m, int n_bump /*height*/, int m_bump /*width*/) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i*m_bump + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	if (i > n_bump-1 || j > m_bump-1)
		return;
		
	float4 bumpIn = read_imagef(Bump, smp, coords); // (bump, mask, label, 0)
	float4 colorIn = read_imagef(RGBMapBump, smp, coords);
	
	if (bumpIn.z == -1.0f) {
		float4 zero1 = {0.0f,-200.0f,-1.0f,0.0f};
		write_imagef(BumpSwap, coords, zero1);
		float4 zero = {0.0f,0.0f,0.0f,0.0f};
		write_imagef(VMapBump, coords, zero);
		write_imagef(RGBMapBumpSwap, coords, zero);
		return;
	}
	
	// Get current point and normal on the template with coefficients
	float nmle[3] = {0.0f,0.0f,0.0f};
	float pt[3] = {0.0f,0.0f,0.0f};
	pt[0] = VerticesBS[6*tid];
	pt[1] = VerticesBS[6*tid + 1];
	pt[2] = VerticesBS[6*tid + 2];
	nmle[0] = VerticesBS[6*tid + 3];
	nmle[1] = VerticesBS[6*tid + 4];
	nmle[2] = VerticesBS[6*tid + 5];
	
	#pragma unroll
	for (int k = 1; k < NB_BS; k++) {
		// This blended normal is not really a normal since it may be not normalized
		nmle[0] = nmle[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 3] * BlendshapeCoeff[k];
		nmle[1] = nmle[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 4] * BlendshapeCoeff[k];
		nmle[2] = nmle[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 5] * BlendshapeCoeff[k];

		pt[0] = pt[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid] * BlendshapeCoeff[k];
		pt[1] = pt[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 1] * BlendshapeCoeff[k];
		pt[2] = pt[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 2] * BlendshapeCoeff[k];
	}

	float4 pt_T = {0.0f,0.0f,0.0f,1.0f};
	pt_T.x = pt[0] * Pose[0] + pt[1] * Pose[4] + pt[2] * Pose[8] + Pose[12];
	pt_T.y = pt[0] * Pose[1] + pt[1] * Pose[5] + pt[2] * Pose[9] + Pose[13];
	pt_T.z = pt[0] * Pose[2] + pt[1] * Pose[6] + pt[2] * Pose[10] + Pose[14];
	
	float4 nmle_T = {0.0f,0.0f,0.0f,0.0f};
	nmle_T.x = nmle[0] * Pose[0] + nmle[1] * Pose[4] + nmle[2] * Pose[8];
	nmle_T.y = nmle[0] * Pose[1] + nmle[1] * Pose[5] + nmle[2] * Pose[9];
	nmle_T.z = nmle[0] * Pose[2] + nmle[1] * Pose[6] + nmle[2] * Pose[10];

	float norm_nT = nmle_T.x*nmle_T.x + nmle_T.y*nmle_T.y + nmle_T.z*nmle_T.z;
	
	
	float2 param = {0.0f, 0.0f};
	// int indx = tid
	
	for (int iter = 0; iter < 1; iter++) {
		////////////////////////////////////////
		///// update distribution for bump nodes
		////////////////////////////////////////
		
		// 1. send message from varNodes to each child (same message for all child)
		
		// 2. update natural parameter vector
		param.x = Prior[2*tid];
		param.y = Prior[2*tid + 1];
		
		for (int ch = 0; ch < 100; ch++) {
			if (Childs[100*tid+ch] == -1) {
				break;
			}
			int indx_ch = Childs[100*tid+ch];
			//printf("indx_ch: %d", indx_ch);
			
			// Get messages from coparents
			float4 v = read_imagef(VMap, smp, (int2){indx_ch - m*convert_int(indx_ch/m), convert_int(indx_ch/m)});
			v.w = 1.0f;
			float4 colorCurr = read_imagef(RGBMap, smp, (int2){indx_ch - m*convert_int(indx_ch/m), convert_int(indx_ch/m)});
			float messages[200];
			float message_color[100];
			int pa = 0;
			for (int curr_pa = 0; curr_pa < 100; curr_pa++) {
				int indx_pa = Parents[100*indx_ch+curr_pa];
				if(indx_pa == -1) {
					break;
				}
				
				if (indx_pa != tid) {
					messages[2*pa] = NaturalParam[2*indx_pa];
					messages[2*pa+1] = NaturalParam[2*indx_pa+1];
					
					float4 color_in = read_imagef(RGBMapBump, smp, (int2) {convert_int(indx_pa/m_bump), indx_pa-m_bump*convert_int(indx_pa/m_bump)});
					float dist_color = 0.0f;
					if ((color_in.x != 0.0f || color_in.y != 0.0f || color_in.z != 0.0f)) {
						dist_color = sqrt((color_in.x - colorCurr.x)*(color_in.x-colorCurr.x) +
										  (color_in.y - colorCurr.y)*(color_in.y-colorCurr.y) +
										  (color_in.z - colorCurr.z)*(color_in.z-colorCurr.z));
					}
					
					message_color[pa] = dist_color;
					
					pa++;
				}
			}
			messages[2*pa] = -1.0f;
			messages[2*pa+1] = -1.0f;
			
			if (v.z == 0.0f) {
				continue;
			}
			
			float prod = 1.0f;
			for (int m = 0; m < 100; m++) {
				if (messages[2*m] == -1.0f) {
					break;
				}
				
				float4 vect = pt_T + messages[2*m]*nmle_T - v;
				prod = prod * ((vect.x*vect.x + vect.y*vect.y + vect.z*vect.z) + 1.0f*message_color[m]);
			}
			
			float sigma = 20000.0f;
			param = param + (float2){sigma*((nmle_T.x*(v.x-pt_T.x) + nmle_T.y*(v.y-pt_T.y) + nmle_T.z*(v.z-pt_T.z)))*prod, -norm_nT*(sigma/2.0f)*prod};
			//printf("param: %f, %f \n", param.x, param.y);
		}
		
		// 3. Compute new expectation
		NaturalParam[2*tid] = -param.x/(2.0f*param.y);
		NaturalParam[2*tid+1] = (param.x*param.x)/(4.0f*param.y*param.y) - 1.0f/(2.0f*param.y);
		
		// sync threads
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	//////////////////////////////////////
	///// Visibility check
	//////////////////////////////////////

	//printf("NaturalParam: %f, %f \n", NaturalParam[2*tid], NaturalParam[2*tid+1]);
	
	float4 pt_N = pt_T + NaturalParam[2*tid]*nmle_T;
	int2 s = (int2){convert_int((pt_N.x/fabs(pt_N.z))*(calib[0]) + (calib[2])),
					n-convert_int((pt_N.y/fabs(pt_N.z))*calib[1] + calib[3])};
	
	if (s.y < 0 || s.x < 0 || s.y > n-1 || s.x > m-1) {
		write_imagef(BumpSwap, coords, bumpIn);
		write_imagef(VMapBump, coords, (float4) {0.0f,0.0f,0.0f,0.0f});
		return;
	}
	
	float4 v = read_imagef(VMap, smp, s);
	v.w = 1.0f;
	if (pt_N.z > v.z + 0.01f && v.z < 0.0f) {
		write_imagef(BumpSwap, coords, (float4) {bumpIn.x, min(-200.0f, bumpIn.y/1.2f), bumpIn.z, bumpIn.w});
		write_imagef(RGBMapBumpSwap, coords, colorIn);
		write_imagef(VMapBump, coords, (float4) {0.0f,0.0f,0.0f,0.0f});
		return;
	}
		
	////////////////////////////////////
	/////// Update Prior
	////////////////////////////////////
	
	float avg = param.x/(-2.0f*param.y);
	float var = max(-100000.0f, param.y);
	Prior[2*tid] = avg*(-2.0f*var);
	Prior[2*tid+1] = var;
	
	write_imagef(BumpSwap, coords, (float4) {avg, var, bumpIn.z, bumpIn.w});
	write_imagef(RGBMapBumpSwap, coords, colorIn);
	pt_N.x = pt[0] + NaturalParam[2*tid]*nmle[0];
	pt_N.y = pt[1] + NaturalParam[2*tid]*nmle[1];
	pt_N.z = pt[2] + NaturalParam[2*tid]*nmle[2];
	//if (var < -200.0f) {
		write_imagef(VMapBump, coords, pt_N);
		//printf("pt_N: %f %f %f\n", pt_N.x, pt_N.y, pt_N.z);
//	} else
//		write_imagef(VMapBump, coords, (float4) {0.0f,0.0f,0.0f,0.0f});
}



