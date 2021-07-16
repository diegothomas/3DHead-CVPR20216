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


bool search (int tid, int2 coords, __constant float *calib, __constant float *Pose, __read_only image2d_t VMap, __read_only image2d_t NMap,
										__global float *VerticesBS, __read_only image2d_t BumpSwap, __constant float *BlendshapeCoeff,
										int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/, 
										double **pt_T, double *s) {
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
		float4 nprev;
		float4 vprev;
		int p_indx[2];	
		float distThres = 0.01;
		float angleThres = 0.8;
		const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
		
		float4 bumpIn = read_imagef(BumpSwap, smp, coords); // (bump, mask, label, 0)
	
		if (bumpIn.y == 0.0f) {
			return false;
		}
			
		float nmle[3] = {0.0f,0.0f,0.0f};
		float pt[3] = {0.0f,0.0f,0.0f};
		float nmleTmp[3] = {0.0f,0.0f,0.0f};
		float ptTmp[3] = {0.0f,0.0f,0.0f};
		float d = bumpIn.x / 1000.0f;
		
		pt[0] = VerticesBS[6*tid];
		pt[1] = VerticesBS[6*tid + 1];
		pt[2] = VerticesBS[6*tid + 2];
		nmleTmp[0] = VerticesBS[6*tid + 3];
		nmleTmp[1] = VerticesBS[6*tid + 4];
		nmleTmp[2] = VerticesBS[6*tid + 5];

		ptTmp[0] = pt[0] + d*nmleTmp[0];
		ptTmp[1] = pt[1] + d*nmleTmp[1];
		ptTmp[2] = pt[2] + d*nmleTmp[2];
		
		nmle[0] = nmleTmp[0] * Pose[0] + nmleTmp[1] * Pose[4] + nmleTmp[2] * Pose[8];
		nmle[1] = nmleTmp[0] * Pose[1] + nmleTmp[1] * Pose[5] + nmleTmp[2] * Pose[9];
		nmle[2] = nmleTmp[0] * Pose[2] + nmleTmp[1] * Pose[6] + nmleTmp[2] * Pose[10];
		
		pt[0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[4] + ptTmp[2] * Pose[8];
		pt[1] = ptTmp[0] * Pose[1] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[9];
		pt[2] = ptTmp[0] * Pose[2] + ptTmp[1] * Pose[6] + ptTmp[2] * Pose[10];
	
		for (int k = 1; k < 28; k++) {
			// This blended normal is not really a normal since it may be not normalized
			nmleTmp[0] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 3];
			nmleTmp[1] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 4];
			nmleTmp[2] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 5];

			ptTmp[0] = VerticesBS[k*6*n_bump*m_bump + 6*tid];
			ptTmp[1] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 1];
			ptTmp[2] = VerticesBS[k*6*n_bump*m_bump + 6*tid + 2];
			
			ptTmp[0] = ptTmp[0] + d*nmleTmp[0];
			ptTmp[1] = ptTmp[1] + d*nmleTmp[1];
			ptTmp[2] = ptTmp[2] + d*nmleTmp[2];
			
			pt_T[k-1][0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[4] + ptTmp[2] * Pose[8];
			pt_T[k-1][1] = ptTmp[0] * Pose[1] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[9];
			pt_T[k-1][2] = ptTmp[0] * Pose[2] + ptTmp[1] * Pose[6] + ptTmp[2] * Pose[10];
			
			pt[0] = pt[0] + BlendshapeCoeff[k] * pt_T[k-1][0];
			pt[1] = pt[1] + BlendshapeCoeff[k] * pt_T[k-1][1];
			pt[2] = pt[2] + BlendshapeCoeff[k] * pt_T[k-1][2];
		}
		
		pt[0] = pt[0] + Pose[12];
		pt[1] = pt[1] + Pose[13];
		pt[2] = pt[2] + Pose[14];
			
		p_indx[0] = min(m_col-1, max(0, convert_int(round((pt[0]/fabs(pt[2]))*calib[0] + calib[2])))); 
		p_indx[1] = n_row - 1 - min(n_row-1, max(0, convert_int(round((pt[1]/fabs(pt[2]))*calib[1] + calib[3])))); 

		nprev = read_imagef(NMap, smp, (int2){p_indx[0], p_indx[1]});
		
		if (nprev.x == 0.0 && nprev.y == 0.0 && nprev.z == 0.0)
			return false;
			
		vprev = read_imagef(VMap, smp, (int2){p_indx[0], p_indx[1]}); // (x,y,z, flag)
		
		float dist = sqrt((vprev.x-pt[0])*(vprev.x-pt[0]) + (vprev.y-pt[1])*(vprev.y-pt[1]) + (vprev.z-pt[2])*(vprev.z-pt[2]));
		if (dist > distThres)
			return false;
		
		float angle = nmle[0]*nprev.x + nmle[1]*nprev.y + nmle[2]*nprev.z;
		if (angle < angleThres)
			return false;

		pt_T[27][0] = convert_double(vprev.x-pt[0]);
		pt_T[27][1] = convert_double(vprev.y-pt[1]);
		pt_T[27][2] = convert_double(vprev.z-pt[2]);
		s[0] = convert_double(dist);
		return true;

}

__kernel void SystemKernel(__read_only image2d_t VMap, __read_only image2d_t NMap,  
						__global float *VerticesBS, __read_only image2d_t BumpSwap, __global double *buf,
						int fact, __constant float *Pose, __constant float *calib, __constant float *BlendshapeCoeff,
						int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/, 
						int BLOCK_SIZE_X, int BLOCK_SIZE_Y) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/

    bool found_coresp = false;
	double weight = 1.0;
	
	double row[28][3];
	double count = 1.0;
	double min_dist;
	
    if ((i*fact) < n_bump && (j*fact) < m_bump)
        found_coresp = search ((i*fact) * m_bump + (j*fact), (int2){j*fact,i*fact}, calib, Pose, VMap, NMap, VerticesBS, BumpSwap, BlendshapeCoeff, n_row, m_col, n_bump, m_bump, row, &min_dist);
	
	// row [0 -> 5] = A^t = [skew(s) | Id(3,3)]^t*n
    if (!found_coresp)
    {
		min_dist = 0.0;
		count = 0.0;
		for (int k = 0; k < 28; k++) {
			row[k][0] = 0.0;
			row[k][1] = 0.0;
			row[k][2] = 0.0;
		}
    }

	////////////// Compute J^t*b ///////////////////////////
	
	__local double smem[THREAD_SIZE*THREAD_SIZE];
	
	int tid_b = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	
    int shift = 0;
	#pragma unroll
	for (int k = 0; k < 27; k++) {
		smem[tid_b] = row[k][0]*row[27][0] + row[k][1]*row[27][1] + row[k][2]*row[27][2];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		reduce(smem, THREAD_SIZE*THREAD_SIZE);
		
		if (tid_b == 0) {			
			buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
		}
	}
	
	// compute current residual
	smem[tid_b] = convert_double(min_dist);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid_b == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
	
	// compute number of matches
	smem[tid_b] = count;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid_b == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
}