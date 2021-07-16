#pragma OPENCL EXTENSION cl_amd_printf : enable
#define	NB_BS 28

__kernel void VmapBumpKernel(__read_only image2d_t Bump, __write_only image2d_t VMapBump,
						__constant float *BlendshapeCoeff, __global float *VerticesBS, __constant float *Pose, int n_bump, int m_bump) {

	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	unsigned int tid = i*m_bump + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	if ( i > n_bump-2 || j > m_bump-2)  {
        return;
    }
	
	float4 bumpIn = read_imagef(Bump, smp, coords); // (bump, mask, label, 0)
	float4 zero = {0.0f,0.0f,0.0f,0.0f};
	if (bumpIn.z == -1.0f) {
		write_imagef(VMapBump, coords, zero);
		return;
	}
	
	float nmle[3] = {0.0f,0.0f,0.0f};
	float pt[3] = {0.0f,0.0f,0.0f};
	pt[0] = VerticesBS[6*tid];
	pt[1] = VerticesBS[6*tid + 1];
	pt[2] = VerticesBS[6*tid + 2];
	nmle[0] = VerticesBS[6*tid + 3];
	nmle[1] = VerticesBS[6*tid + 4];
	nmle[2] = VerticesBS[6*tid + 5];
	
	for (int k = 1; k < NB_BS; k++) {
		// This blended normal is not really a normal since it may be not normalized
		nmle[0] = nmle[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 3] * BlendshapeCoeff[k];
		nmle[1] = nmle[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 4] * BlendshapeCoeff[k];
		nmle[2] = nmle[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 5] * BlendshapeCoeff[k];

		pt[0] = pt[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid] * BlendshapeCoeff[k];
		pt[1] = pt[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 1] * BlendshapeCoeff[k];
		pt[2] = pt[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 2] * BlendshapeCoeff[k];
		//printf("pt: %f %f %f\n", VerticesBS[k*6*n_bump*m_bump + 6*tid], VerticesBS[k*6*n_bump*m_bump + 6*tid + 1], VerticesBS[k*6*n_bump*m_bump + 6*tid + 2]);
	}
	
	float pt_T[3];
	pt_T[0] = pt[0] * Pose[0] + pt[1] * Pose[4] + pt[2] * Pose[8] + Pose[12];
	pt_T[1] = pt[0] * Pose[1] + pt[1] * Pose[5] + pt[2] * Pose[9] + Pose[13];
	pt_T[2] = pt[0] * Pose[2] + pt[1] * Pose[6] + pt[2] * Pose[10] + Pose[14];
	
	float nmle_T[3];
	nmle_T[0] = nmle[0] * Pose[0] + nmle[1] * Pose[4] + nmle[2] * Pose[8];
	nmle_T[1] = nmle[0] * Pose[1] + nmle[1] * Pose[5] + nmle[2] * Pose[9];
	nmle_T[2] = nmle[0] * Pose[2] + nmle[1] * Pose[6] + nmle[2] * Pose[10];
	
	float fact_BP = 1000.0f;
	float bum_val = bumpIn.x;
	float d = bum_val/fact_BP;
	
	//float4 VMapInn = {pt_T[0] + d*nmle_T[0], pt_T[1] + d*nmle_T[1], pt_T[2] + d*nmle_T[2], 0.0f};
	float4 VMapInn = {pt[0] + d*nmle[0], pt[1] + d*nmle[1], pt[2] + d*nmle[2], bumpIn.y};
	//if (tid == 36079)
	//	printf("pt: %f %f %f %d\n", pt[0], pt[1], pt[2], tid);
	
	write_imagef(VMapBump, coords, VMapInn);
	
}