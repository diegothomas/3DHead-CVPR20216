#pragma OPENCL EXTENSION cl_amd_printf : enable

#define THREAD_SIZE 8

bool searchLandMark (int l_idx, __constant float *Pose, __read_only image2d_t VMap, __read_only image2d_t NMap,
										__read_only image2d_t VMapBump, __read_only image2d_t NMapBump, 
										__constant int *landmarksBump, __constant float *landmarks, 
										int n_bump, int m_bump, float *n, float *d, float *s)
										{
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
		const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
		float4 ncurr;
		float ncurr_cp[3];
		float4 nprev;
		float4 vcurr;	
		float vcurr_cp[3];
		float4 vprev;
		float distThres = 0.03f;
		float angleThres = 0.6f;
		
		int idx_i = landmarksBump[l_idx];
		int idx_j = landmarksBump[43 + l_idx];
		
		//printf("Landmark %d: %d, %d \n", l_idx, idx_i, idx_j);
		
		if (idx_i < 0 || idx_i > n_bump - 1 || idx_j < 0 || idx_j > m_bump - 1)
			return false;
			
		ncurr = read_imagef(NMapBump, smp, (int2){idx_j, idx_i});

		if (ncurr.x == 0.0 && ncurr.y == 0.0 && ncurr.z == 0.0)
			return false;
			
		if (ncurr.z < 0.0)
			return false;
		
		vcurr = read_imagef(VMapBump, smp, (int2){idx_j, idx_i});

		vcurr_cp[0] = Pose[0]*vcurr.x + Pose[4]*vcurr.y + Pose[8]*vcurr.z + Pose[12]; //Rcurr is row major
		vcurr_cp[1] = Pose[1]*vcurr.x + Pose[5]*vcurr.y + Pose[9]*vcurr.z + Pose[13];
		vcurr_cp[2] = Pose[2]*vcurr.x + Pose[6]*vcurr.y + Pose[10]*vcurr.z + Pose[14];

		ncurr_cp[0] = Pose[0]*ncurr.x + Pose[4]*ncurr.y + Pose[8]*ncurr.z; //Rcurr is row major
		ncurr_cp[1] = Pose[1]*ncurr.x + Pose[5]*ncurr.y + Pose[9]*ncurr.z;
		ncurr_cp[2] = Pose[2]*ncurr.x + Pose[6]*ncurr.y + Pose[10]*ncurr.z;

		int p_u = convert_int(round(landmarks[l_idx]));		
		int p_v = convert_int(round(landmarks[43 + l_idx]));
		
		if (p_u == 0 && p_v == 0)
			return false;
		
		nprev = read_imagef(NMap, smp, (int2){p_u, p_v});
		
		if (nprev.x == 0.0 && nprev.y == 0.0 && nprev.z == 0.0)
			return false;
			
		vprev = read_imagef(VMap, smp, (int2){p_u, p_v}); // (x,y,z, flag)
		
		float dist = sqrt((vprev.x-vcurr_cp[0])*(vprev.x-vcurr_cp[0]) + (vprev.y-vcurr_cp[1])*(vprev.y-vcurr_cp[1]) + (vprev.z-vcurr_cp[2])*(vprev.z-vcurr_cp[2]));
		if (dist > distThres)
			return false;
		
		/*float angle = ncurr_cp[0]*nprev.x + ncurr_cp[1]*nprev.y + ncurr_cp[2]*nprev.z;
		if (angle < angleThres)
			return false;*/

		n[0] =  ncurr_cp[0]; n[1] =  ncurr_cp[1]; n[2] =  ncurr_cp[2];
		d[0] =  vcurr_cp[0]; d[1] =  vcurr_cp[1]; d[2] =  vcurr_cp[2];
		s[0] =  vprev.x; s[1] = vprev.y; s[2] = vprev.z; s[3] = - vprev.z;
		
		//printf("Landmark %d: %f %f %f\n %f %f %f \n", l_idx, vprev.x, vprev.y, vprev.z, vcurr_cp[0], vcurr_cp[1], vcurr_cp[2]);
	
		return true;
}

bool searchGauss (int2 coords, __constant float *calib, __constant float *Pose, __read_only image2d_t VMap, __read_only image2d_t NMap,
										__read_only image2d_t VMapBump, __read_only image2d_t NMapBump, 
										int n_row, int m_col, 
										float *n, float *d, float *s) {
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev

        float4 ncurr;
		float ncurr_cp[3];
		float4 nprev;
		float4 vcurr;	
		float vcurr_cp[3];
		float4 vprev;
		int p_indx[2];	
		float distThres = 0.03;
		float angleThres = 0.6;
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

		n[0] =  ncurr_cp[0]; n[1] =  ncurr_cp[1]; n[2] =  ncurr_cp[2];
		d[0] =  vcurr_cp[0]; d[1] =  vcurr_cp[1]; d[2] =  vcurr_cp[2];
		s[0] =  vprev.x; s[1] =  vprev.y; s[2] =  vprev.z; s[3] = - vprev.z;
		s[4] =  max(1.0f, vcurr.w/10.0f);

		return true;

}

// Replace __global by __constant ?? 

//label contains 4  hannel unsigned char value. the last channel defines pixels of the front face
__kernel void GICPKernel(__read_only image2d_t VMap, __read_only image2d_t NMap,  
						__read_only image2d_t VMapBump, __read_only image2d_t NMapBump,__global float *buf, __read_only image2d_t label,
						int fact, __constant float *Pose, __constant float *calib, __global int* nbMatches,
						__constant int *landmarksBump, __constant float *landmarks, 
						int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	float n[3], d[3], s[5];
    bool found_coresp = false;
	float weight = 1.0f;
	
	uint4 lab = read_imageui(label, smp, (int2){j*fact,i*fact});
	
    if ((i*fact) < n_bump && (j*fact) < m_bump) {
        found_coresp = searchGauss ((int2){j*fact,i*fact}, calib, Pose, VMap, NMap, VMapBump, NMapBump, n_row, m_col, n, d, s);
		weight = s[4];
	}
	
	if ((i*fact) == n_bump && j < 43) {
        found_coresp = searchLandMark (j, Pose, VMap, NMap, VMapBump, NMapBump, landmarksBump, landmarks, n_bump, m_bump, n, d, s);
		weight = 10.0f;
	}
	
    if (!found_coresp)
		return;
	
	float row[7];
	float rowY[7];
	float rowZ[7];
	float JD[18];
	float JRot[18];
	float min_dist = 0.0;
	int indx_buff;
	int indx_buffY;
	int indx_buffZ;

	// row [0 -> 5] = A^t = [skew(s) | Id(3,3)]^t*n
	//weight = 1.0;//0.0012/(0.0012 + 0.0019*(s[3]-0.4)*(s[3]-0.4));
	
	JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0;		JD[12] = 2.0*d[2];	JD[15] = -2.0*d[1];
	JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -2.0*d[2]; JD[13] = 0.0;		JD[16] = 2.0*d[0];
	JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = 2.0*d[1];	JD[14] = -2.0*d[0]; JD[17] = 0.0;

	if ((i*fact) < n_bump && (j*fact) < m_bump) {
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
		
	
		min_dist = sqrt((s[0]-d[0])*(s[0]-d[0]) + (s[1]-d[1])*(s[1]-d[1]) + (s[2]-d[2])*(s[2]-d[2]));
		
		indx_buff = atomic_inc(nbMatches);
		
		#pragma unroll
		for (int k = 0; k < 7; k++) {
			buf[8*indx_buff + k] = row[k];
		}
		buf[8*indx_buff + 7] = min_dist;
	} else {
		// Landmark
		row[0] = weight*(JD[0]);
		row[1] = weight*(JD[3]);
		row[2] = weight*(JD[6]);
		row[3] = weight*(JD[9]);
		row[4] = weight*(JD[12]);
		row[5] = weight*(JD[15]);
		row[6] = weight*(s[0]-d[0]);
		
		rowY[0] = weight*(JD[1]);
		rowY[1] = weight*(JD[4]);
		rowY[2] = weight*(JD[7]);
		rowY[3] = weight*(JD[10]);
		rowY[4] = weight*(JD[13]);
		rowY[5] = weight*(JD[16]);
		rowY[6] = weight*(s[1]-d[1]);
		
		rowZ[0] = weight*(JD[2]);
		rowZ[1] = weight*(JD[5]);
		rowZ[2] = weight*(JD[8]);
		rowZ[3] = weight*(JD[11]);
		rowZ[4] = weight*(JD[14]);
		rowZ[5] = weight*(JD[17]);
		rowZ[6] = weight*(s[2]-d[2]);
		
		indx_buff = atomic_inc(nbMatches);
		indx_buffY = atomic_inc(nbMatches);
		indx_buffZ = atomic_inc(nbMatches);
		
		#pragma unroll
		for (int k = 0; k < 7; k++) {
			buf[8*indx_buff + k] = row[k];
			buf[8*indx_buffY + k] = rowY[k];
			buf[8*indx_buffZ + k] = rowZ[k];
		}
		buf[8*indx_buff + 7] = (s[0]-d[0]);
		buf[8*indx_buffY + 7] = (s[1]-d[1]);
		buf[8*indx_buffZ + 7] = (s[2]-d[2]);
	}
}