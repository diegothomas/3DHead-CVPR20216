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


bool searchGauss (int2 coords, __constant float *calib, __constant float *Pose, __read_only image2d_t VMap, __read_only image2d_t NMap,
										__read_only image2d_t VMapBump, __read_only image2d_t NMapBump, 
										int n_row, int m_col, 
										double *n, double *d, double *s) {
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev

        float4 ncurr;
		float ncurr_cp[3];
		float4 nprev;
		float4 vcurr;	
		float vcurr_cp[3];
		float4 vprev;
		int p_indx[2];	
		float distThres = 0.01;
		float angleThres = 0.8;
		const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
		
		ncurr = read_imagef(NMapBump, smp, coords);

		if (ncurr.x == 0.0 && ncurr.y == 0.0 && ncurr.z == 0.0)
			return false;
		
		vcurr = read_imagef(VMapBump, smp, coords);

		vcurr_cp[0] = Pose[0]*vcurr.x + Pose[4]*vcurr.y + Pose[8]*vcurr.z + Pose[12]; //Rcurr is row major
		vcurr_cp[1] = Pose[1]*vcurr.x + Pose[5]*vcurr.y + Pose[9]*vcurr.z + Pose[13];
		vcurr_cp[2] = Pose[2]*vcurr.x + Pose[6]*vcurr.y + Pose[10]*vcurr.z + Pose[14];

		ncurr_cp[0] = Pose[0]*ncurr.x + Pose[4]*ncurr.y + Pose[8]*ncurr.z; //Rcurr is row major
		ncurr_cp[1] = Pose[1]*ncurr.x + Pose[5]*ncurr.y + Pose[9]*ncurr.z;
		ncurr_cp[2] = Pose[2]*ncurr.x + Pose[6]*ncurr.y + Pose[10]*ncurr.z;
			
		p_indx[0] = min(m_col-1, max(0, convert_int(round((vcurr_cp[0]/fabs(vcurr_cp[2]))*calib[0] + calib[2])))); 
		p_indx[1] = n_row - 1 - min(n_row-1, max(0, convert_int(round((vcurr_cp[1]/fabs(vcurr_cp[2]))*calib[1] + calib[3])))); 

		nprev = read_imagef(NMap, smp, (int2){p_indx[0], p_indx[1]});
		
		if (nprev.x == 0.0 && nprev.y == 0.0 && nprev.z == 0.0)
			return false;
			
		vprev = read_imagef(VMap, smp, (int2){p_indx[0], p_indx[1]}); // (x,y,z, flag)
		
		float dist = sqrt((vprev.x-vcurr_cp[0])*(vprev.x-vcurr_cp[0]) + (vprev.y-vcurr_cp[1])*(vprev.y-vcurr_cp[1]) + (vprev.z-vcurr_cp[2])*(vprev.z-vcurr_cp[2]));
		if (dist > distThres)
			return false;
		
		float angle = ncurr_cp[0]*nprev.x + ncurr_cp[1]*nprev.y + ncurr_cp[2]*nprev.z;
		if (angle < angleThres)
			return false;

		n[0] =  convert_double(ncurr_cp[0]); n[1] =  convert_double(ncurr_cp[1]); n[2] =  convert_double(ncurr_cp[2]);
		d[0] =  convert_double(vcurr_cp[0]); d[1] =  convert_double(vcurr_cp[1]); d[2] =  convert_double(vcurr_cp[2]);
		s[0] =  convert_double(vprev.x); s[1] =  convert_double(vprev.y); s[2] =  convert_double(vprev.z); s[3] = - convert_double(vprev.z);

		return true;

}

// Replace __global by __constant ?? 

__kernel void GICPKernel(__read_only image2d_t VMap, __read_only image2d_t NMap,  
						__read_only image2d_t VMapBump, __read_only image2d_t NMapBump,__global double *buf, __read_only image2d_t label,
						int fact, __constant float *Pose, __constant float *calib, 
						int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/, 
						int BLOCK_SIZE_X, int BLOCK_SIZE_Y) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	double n[3], d[3], s[4];
    bool found_coresp = false;
	double weight = 1.0;
	
	uint4 lab = read_imageui(label, smp, (int2){j*fact,i*fact});
	
    if ((i*fact) < n_bump && (j*fact) < m_bump /*&& lab.w == 255*/)
        found_coresp = searchGauss ((int2){j*fact,i*fact}, calib, Pose, VMap, NMap, VMapBump, NMapBump, n_row, m_col, n, d, s);
	
	double row[7];
	row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.0;
	double JD[18];
	double JRot[18];
	double min_dist = 0.0;
	double count = 0.0;

	// row [0 -> 5] = A^t = [skew(s) | Id(3,3)]^t*n
    if (found_coresp)
    {
		weight = 1.0;//0.0012/(0.0012 + 0.0019*(s[3]-0.4)*(s[3]-0.4));
		
		JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0;		JD[12] = 2.0*d[2];	JD[15] = -2.0*d[1];
		JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -2.0*d[2]; JD[13] = 0.0;		JD[16] = 2.0*d[0];
		JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = 2.0*d[1];	JD[14] = -2.0*d[0]; JD[17] = 0.0;

		JRot[0] = 0.0; JRot[3] = 0.0; JRot[6] = 0.0;	JRot[9] = 0.0;			JRot[12] = 2.0*n[2];	JRot[15] = -2.0*n[1];
		JRot[1] = 0.0; JRot[4] = 0.0; JRot[7] = 0.0;	JRot[10] = -2.0*n[2];	JRot[13] = 0.0;			JRot[16] = 2.0*n[0];
		JRot[2] = 0.0; JRot[5] = 0.0; JRot[8] = 0.0;	JRot[11] = 2.0*n[1];	JRot[14] = -2.0*n[0];	JRot[17] = 0.0;

		row[0] = weight*(-(n[0] * JD[0] + n[1] * JD[1] + n[2] * JD[2]) + JRot[0] * (s[0] - d[0]) + JRot[1] * (s[1] - d[1]) + JRot[2] * (s[2] - d[2]));
		row[1] = weight*(-(n[0] * JD[3] + n[1] * JD[4] + n[2] * JD[5]) + JRot[3] * (s[0] - d[0]) + JRot[4] * (s[1] - d[1]) + JRot[5] * (s[2] - d[2]));
		row[2] = weight*(-(n[0] * JD[6] + n[1] * JD[7] + n[2] * JD[8]) + JRot[6] * (s[0] - d[0]) + JRot[7] * (s[1] - d[1]) + JRot[8] * (s[2] - d[2]));
		row[3] = weight*(-(n[0] * JD[9] + n[1] * JD[10] + n[2] * JD[11]) + JRot[9] * (s[0] - d[0]) + JRot[10] * (s[1] - d[1]) + JRot[11] * (s[2] - d[2]));
		row[4] = weight*(-(n[0] * JD[12] + n[1] * JD[13] + n[2] * JD[14]) + JRot[12] * (s[0] - d[0]) + JRot[13] * (s[1] - d[1]) + JRot[14] * (s[2] - d[2]));
		row[5] = weight*(-(n[0] * JD[15] + n[1] * JD[16] + n[2] * JD[17]) + JRot[15] * (s[0] - d[0]) + JRot[16] * (s[1] - d[1]) + JRot[17] * (s[2] - d[2]));

		row[6] = -weight*(n[0]*(s[0]-d[0]) + n[1]*(s[1]-d[1]) + n[2]*(s[2]-d[2]));
		//min_dist = fabs(weight*(n[0]*(s[0]-d[0]) + n[1]*(s[1]-d[1]) + n[2]*(s[2]-d[2])));
		min_dist = convert_double(sqrt((s[0]-d[0])*(s[0]-d[0]) + (s[1]-d[1])*(s[1]-d[1]) + (s[2]-d[2])*(s[2]-d[2])));
		count = 1.0;
    }

	////////////// Compute A^t*A and A^t*b ///////////////////////////
	
	__local double smem[THREAD_SIZE*THREAD_SIZE];

	int tid = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	
    int shift = 0;
    for (int k = 0; k < 6; ++k)        //rows
    {
        #pragma unroll
        for (int l = k; l < 7; ++l)          // cols + b
        {
			smem[tid] = row[k] * row[l];
			barrier(CLK_LOCAL_MEM_FENCE);
						
			reduce(smem, THREAD_SIZE*THREAD_SIZE);
			
			if (tid == 0) {			
				buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
			}
		}
    }
	
	// compute current residual
	smem[tid] = min_dist;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
	
	// compute number of matches
	smem[tid] = count;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
}