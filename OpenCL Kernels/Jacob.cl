#pragma OPENCL EXTENSION cl_amd_printf : enable

#define THREAD_SIZE 8

void reduce(__local double* buffer, int CTA_SIZE)
{
	int tid = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	double val =  buffer[tid];

	if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  128) { if (tid <  64) buffer[tid] = val = val + buffer[tid +  64]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  64) { if (tid <  32) buffer[tid] = val = val + buffer[tid +  32]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  32) { if (tid <  16) buffer[tid] = val = val + buffer[tid +  16]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  16) { if (tid <  8) buffer[tid] = val = val + buffer[tid +  8]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  8) { if (tid <  4) buffer[tid] = val = val + buffer[tid +  4]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  4) { if (tid <  2) buffer[tid] = val = val + buffer[tid +  2]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE >=  2) { if (tid == 0) buffer[tid] = val = val + buffer[tid +  1]; barrier(CLK_LOCAL_MEM_FENCE); }
}

__kernel void JacobiKernel(__read_only image2d_t BumpSwap, 
						__global float *VerticesBS, 
						__global double *buf,
						__constant float *Pose, 
						int n_bump /*height*/, int m_bump /*width*/,
						int BLOCK_SIZE_X, int BLOCK_SIZE_Y) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i*m_bump + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	if (i > n_bump-1 || j > m_bump-1)
		return;
		
	float4 bumpIn = read_imagef(BumpSwap, smp, coords); // (bump, mask, label, 0)
	
	bool valid = true;
	if (bumpIn.y == 0.0f) {
		valid = false;
	}
		
	float nmle[3] = {0.0f,0.0f,0.0f};
	float pt[3] = {0.0f,0.0f,0.0f};
	float pt_T[48][3];
	float d = bumpIn.x / 1000.0f;
	
	if (valid){
		for (int k = 1; k < 28; k++) {
			// This blended normal is not really a normal since it may be not normalized
			nmle[0] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 3];
			nmle[1] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 4];
			nmle[2] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 5];

			pt[0] = VerticesBS[k*6*n_bump*m_bump + 6*tid];
			pt[1] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 1];
			pt[2] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 2];
			
			pt[0] = pt[0] + d*nmle[0];
			pt[1] = pt[1] + d*nmle[1];
			pt[2] = pt[2] + d*nmle[2];
			
			pt_T[k-1][0] = pt[0] * Pose[0] + pt[1] * Pose[4] + pt[2] * Pose[8];
			pt_T[k-1][1] = pt[0] * Pose[1] + pt[1] * Pose[5] + pt[2] * Pose[9];
			pt_T[k-1][2] = pt[0] * Pose[2] + pt[1] * Pose[6] + pt[2] * Pose[10];
			
		}
	} else {
		for (int k = 0; k < 27; k++) {
			pt_T[k][0] = 0.0;
			pt_T[k][1] = 0.0;
			pt_T[k][2] = 0.0;
		}
	}

	////////////// Compute J^t*J ///////////////////////////
	
	__local double smem[THREAD_SIZE*THREAD_SIZE];
	
	int tid_b = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	
    int shift = 0;
	#pragma unroll
	for (int k = 1; k < 28; k++) {
		for (int l = k; l < 28; l++) {
			if (valid) {
				smem[tid_b] = convert_double(pt_T[k-1][0]*pt_T[l-1][0] + pt_T[k-1][1]*pt_T[l-1][1] + pt_T[k-1][2]*pt_T[l-1][2]);
			} else {
				smem[tid_b] = 0.0;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			
			reduce(smem, THREAD_SIZE*THREAD_SIZE);
			
			if (tid_b == 0) {			
				buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
			}
		}
	}
}