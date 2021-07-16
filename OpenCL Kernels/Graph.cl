#pragma OPENCL EXTENSION cl_amd_printf : enable

#define	NB_BS 28

__kernel void GraphKernel(__read_only image2d_t Bump, __read_only image2d_t RGBMapBump,
						__read_only image2d_t VMap, __read_only image2d_t NMap, __read_only image2d_t RGBMap,
						__global int *Childs, __global int *Parents,
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

	//printf("Landmark %d: %d, %d \n", l_idx, idx_i, idx_j);
	if (bumpIn.z == -1.0f) {
		Childs[100*tid] = -1;
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
	
	//printf("pt %f, %f, %f\n", pt_T.x, pt_T.y, pt_T.z);
	//printf("nmle_T %f, %f, %f\n", nmle_T.x, nmle_T.y, nmle_T.z);

	if (nmle_T.z < 0.2f) {
		Childs[100*tid] = -1;
		return;
	}
	
	//printf("bump_in %f, %f\n", bumpIn.x, bumpIn.y);
	float range = 0.3f/sqrt(-2.0f*bumpIn.y);

	float4 p1 = pt_T + (bumpIn.x-range) * nmle_T;
	float4 p2 = pt_T + (bumpIn.x+range) * nmle_T;
	
	// project the point onto the depth image
	float2 s1 = (float2) {(p1.x/fabs(p1.z))*(calib[0]) + (calib[2]), convert_float(n)-((p1.y/fabs(p1.z))*(calib[1])+(calib[3]))};
	float2 s2 = (float2) {(p2.x/fabs(p2.z))*(calib[0]) + (calib[2]), convert_float(n)-((p2.y/fabs(p2.z))*(calib[1])+(calib[3]))};
	
	//printf("s1 %f, %f\n", s1.x, s1.y);
	//printf("s2 %f, %f\n", s2.x, s2.y);
	
	float lengthRay = sqrt((s1.x-s2.x)*(s1.x-s2.x) + (s1.y-s2.y)*(s1.y-s2.y));
	
	float2 dir = (float2) {(s2.x - s1.x)/lengthRay, (s2.y-s1.y)/lengthRay};
	
	if (lengthRay < 1.0f) {
		lengthRay = 1.0f;
	}

	//printf("lengthRay %f\n", lengthRay);
	
	// add all pixels in the window as childs
	int count = 0;
	int kPrev = -1;
	int lPrev = -1;
	for (int lambda = 0; lambda < convert_int(round(lengthRay))-1; lambda++) {
		int k = convert_int(round(s1.y+convert_float(lambda)*dir.y));
		int l = convert_int(round(s1.x+convert_float(lambda)*dir.x));
	
		if (k == kPrev && l == lPrev) {
			continue;
		}
	
		kPrev = k;
		lPrev = l;
		
		int2 coordsCurr = (int2){l, k};
		
		float4 v = read_imagef(VMap, smp, coordsCurr);
		v.w = 1.0f;
		float4 ncurr = read_imagef(NMap, smp, coordsCurr);
		ncurr.w = 0.0f;
		float4 colorCurr = read_imagef(RGBMap, smp, coordsCurr);
			
		float dist_color = 0.0f;
		if ((colorCurr.x != 0.0f || colorCurr.y != 0.0f || colorCurr.z != 0.0f)) {
			dist_color = sqrt((colorIn.x - colorCurr.x)*(colorIn.x-colorCurr.x) +
							  (colorIn.y - colorCurr.y)*(colorIn.y-colorCurr.y) +
							  (colorIn.z - colorCurr.z)*(colorIn.z-colorCurr.z));
		}
			
		float dist_proj = dot(v-pt_T, nmle_T);
		/*printf("v: %f %f %f %f\n", v.x, v.y, v.z, v.w);
		printf("pt_T: %f %f %f %f\n", pt_T.x, pt_T.y, pt_T.z, pt_T.w);
		printf("nmle_T: %f %f %f %f\n", nmle_T.x, nmle_T.y, nmle_T.z, nmle_T.w);
		printf("dist_proj: %f\n", dist_proj);*/
		float dist = sqrt(dot(v-pt_T - dist_proj*nmle_T, v-pt_T - dist_proj*nmle_T));
		//printf("dist: %f\n", dist);
		float angle = dot(nmle_T, ncurr);
			
		if (dist > 0.01f /*|| angle < 0.4f || dist_color > 0.3f*/ || count == 99) {
			continue;
		}
		
		Childs[100*tid+count] = k*m + l;
		count++;
		//printf("count: %d\n", count);
		for (int countPar = 0; countPar < 100; countPar++) {
			if (Parents[100*(k*m + l)+countPar] == -1) {
				Parents[100*(k*m + l)+countPar] = tid;
				Parents[100*(k*m + l)+countPar+1] = -1;
				break;
			}
		}
	}
	Childs[100*tid+count] = -1;
	//printf("count: %d\n", count);
	
}