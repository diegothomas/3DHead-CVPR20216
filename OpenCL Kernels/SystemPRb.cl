#pragma OPENCL EXTENSION cl_amd_printf : enable

#define THREAD_SIZE 8
#define	NB_BS 28

bool searchLandMark (int l_idx, __constant float *calib, __constant float *Pose, __read_only image2d_t VMap,
					__global float *VerticesBS, __read_only image2d_t Bump, __constant float *BlendshapeCoeff,
					__constant int *landmarksBump, __constant float *landmarks, 
					int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/, 
					float *tmpX, float *tmpY, float *tmpZ) {
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
		float w = 10.0f;
		const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
		float4 vprev;
		
		int idx_i = landmarksBump[l_idx];
		int idx_j = landmarksBump[43 + l_idx];
		
		//printf("Landmark %d: %d, %d", l_idx, idx_i, idx_j);
		
		if (idx_i < 1 || idx_i > n_bump - 1 || idx_j < 1 || idx_j > m_bump - 1)
			return false;
		
		float4 bumpIn = read_imagef(Bump, smp, (int2){idx_j, idx_i}); // (bump, mask, label, 0)
	
		if (bumpIn.y == 0.0f) {
			return false;
		}
		
		float d = bumpIn.x / 1000.0f;
		
		int tid = idx_i*m_bump + idx_j;
		
		int p_u = convert_int(round(landmarks[l_idx]));		
		int p_v = convert_int(round(landmarks[43 + l_idx]));
		
		if ((p_u == 0 && p_v == 0) || p_u < 0 || p_v < 0) 
			return false;
		
		float pt_T[3] = {0.0f,0.0f,0.0f};
		float pt[3] = {0.0f,0.0f,0.0f};
		float nmle[3] = {0.0f,0.0f,0.0f};
		float nmleTmp[3] = {0.0f,0.0f,0.0f};
		
		pt[0] = VerticesBS[6*tid];
		pt[1] = VerticesBS[6*tid + 1];
		pt[2] = VerticesBS[6*tid + 2];
		nmleTmp[0] = VerticesBS[6*tid + 3];
		nmleTmp[1] = VerticesBS[6*tid + 4];
		nmleTmp[2] = VerticesBS[6*tid + 5];
		
		pt_T[0] = pt[0] + d*nmleTmp[0];
		pt_T[1] = pt[1] + d*nmleTmp[1];
		pt_T[2] = pt[2] + d*nmleTmp[2];
		
		nmle[0] = nmleTmp[0] * Pose[0] + nmleTmp[1] * Pose[4] + nmleTmp[2] * Pose[8];
		nmle[1] = nmleTmp[0] * Pose[1] + nmleTmp[1] * Pose[5] + nmleTmp[2] * Pose[9];
		nmle[2] = nmleTmp[0] * Pose[2] + nmleTmp[1] * Pose[6] + nmleTmp[2] * Pose[10];
		// Test if normal oriented backward
		if (nmle[2] < 0.3f)
			return false;
		
		pt[0] = pt_T[0] * Pose[0] + pt_T[1] * Pose[4] + pt_T[2] * Pose[8];
		pt[1] = pt_T[0] * Pose[1] + pt_T[1] * Pose[5] + pt_T[2] * Pose[9];
		pt[2] = pt_T[0] * Pose[2] + pt_T[1] * Pose[6] + pt_T[2] * Pose[10];
		
		#pragma unroll
		for (int k = 1; k < NB_BS; k++) {
			//(f((bj-bo) + d(nj-n0))
			pt_T[0] = VerticesBS[k * 6 * n_bump*m_bump + 6 * tid] + d * VerticesBS[k * 6 * n_bump*m_bump + 6 * tid + 3];
			pt_T[1] = VerticesBS[k * 6 * n_bump*m_bump + 6 * tid + 1] + d * VerticesBS[k * 6 * n_bump*m_bump + 6 * tid + 4];
			pt_T[2] = VerticesBS[k * 6 * n_bump*m_bump + 6 * tid + 2] + d * VerticesBS[k * 6 * n_bump*m_bump + 6 * tid + 5];
			
			tmpX[k - 1] = pt_T[0] * Pose[0] + pt_T[1] * Pose[4] + pt_T[2] * Pose[8];
			tmpY[k - 1] = pt_T[0] * Pose[1] + pt_T[1] * Pose[5] + pt_T[2] * Pose[9];
			tmpZ[k - 1] = pt_T[0] * Pose[2] + pt_T[1] * Pose[6] + pt_T[2] * Pose[10];
			
			pt[0] = pt[0] + BlendshapeCoeff[k] * tmpX[k-1];
			pt[1] = pt[1] + BlendshapeCoeff[k] * tmpY[k-1];
			pt[2] = pt[2] + BlendshapeCoeff[k] * tmpZ[k-1];
			
			tmpX[k - 1] = w*(tmpX[k - 1]);
			tmpY[k - 1] = w*(tmpY[k - 1]);
			tmpZ[k - 1] = w*(tmpZ[k - 1]);
			
		}
		pt[0] = pt[0] + Pose[12];
		pt[1] = pt[1] + Pose[13];
		pt[2] = pt[2] + Pose[14];
		
		vprev = read_imagef(VMap, smp, (int2){p_u, p_v}); // (x,y,z, flag)
		
		float dist = sqrt((pt[0] - vprev.x)*(pt[0] - vprev.x) + (pt[1] - vprev.y)*(pt[1] - vprev.y) + (pt[2] - vprev.z)*(pt[2] - vprev.z));
		if (dist > 0.03f) 
			return false;
		
		tmpX[NB_BS-1] = -w*(pt[0] - vprev.x);
		tmpY[NB_BS-1] = -w*(pt[1] - vprev.y);
		tmpZ[NB_BS-1] = -w*(pt[2] - vprev.z);
	
		return true;
}

bool search (int tid, int2 coords, __constant float *calib, __constant float *Pose, __read_only image2d_t VMap, __read_only image2d_t NMap, __read_only image2d_t NMapBump,
										__global float *VerticesBS, __read_only image2d_t Bump, __constant float *BlendshapeCoeff,
										__read_only image2d_t LabelsMask, 
										int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/, 
										float *tmpX, float *tmpY, float *tmpZ) {
		// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
		float4 nprev;
		float4 vprev;
		int p_indx[2];	
		float distThres = 0.01f;
		float angleThres = 0.7f;
		const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
		
		float4 bumpIn = read_imagef(Bump, smp, coords); // (bump, mask, label, 0)
		uint4 labelIn = read_imageui(LabelsMask, smp, coords); // in the w channel is the flag for the frontface pixels (the ones that change with expression)
		float w = 1.0f;
		float4 nBump = read_imagef(NMapBump, smp, coords);
	
		if (bumpIn.y == 0.0f || labelIn.w == 0 || (nBump.x == 0.0f && nBump.y == 0.0f && nBump.z == 0.0f)) {
			return false;
		}
		
		float nmleBump[3];
		nmleBump[0] = nBump.x * Pose[0] + nBump.y * Pose[4] + nBump.z * Pose[8];
		nmleBump[1] = nBump.x * Pose[1] + nBump.y * Pose[5] + nBump.z * Pose[9];
		nmleBump[2] = nBump.x * Pose[2] + nBump.y * Pose[6] + nBump.z * Pose[10];
			
		float nmle[3] = {0.0f,0.0f,0.0f};
		float pt[3] = {0.0f,0.0f,0.0f};
		float pt_ref[3] = {0.0f,0.0f,0.0f};
		float pt_TB[27][3];
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
		
		if (nmleBump[2] < 0.0f)
			return false;
		
		pt[0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[4] + ptTmp[2] * Pose[8];
		pt[1] = ptTmp[0] * Pose[1] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[9];
		pt[2] = ptTmp[0] * Pose[2] + ptTmp[1] * Pose[6] + ptTmp[2] * Pose[10];
		pt_ref[0] = pt[0] + Pose[12];
		pt_ref[1] = pt[1] + Pose[13];
		pt_ref[2] = pt[2] + Pose[14];
		
		//#pragma unroll
		for (int k = 1; k < NB_BS; k++) {
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
			
			pt_TB[k-1][0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[4] + ptTmp[2] * Pose[8];
			pt_TB[k-1][1] = ptTmp[0] * Pose[1] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[9];
			pt_TB[k-1][2] = ptTmp[0] * Pose[2] + ptTmp[1] * Pose[6] + ptTmp[2] * Pose[10];
			//tmpX[k - 1] = w*(ptTmp[0] * Pose[0] + ptTmp[1] * Pose[4] + ptTmp[2] * Pose[8]);
			//tmpY[k - 1] = w*(ptTmp[0] * Pose[1] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[9]);
			//tmpZ[k - 1] = w*(ptTmp[0] * Pose[2] + ptTmp[1] * Pose[6] + ptTmp[2] * Pose[10]);
			
			pt[0] = pt[0] + BlendshapeCoeff[k] * pt_TB[k-1][0];
			pt[1] = pt[1] + BlendshapeCoeff[k] * pt_TB[k-1][1];
			pt[2] = pt[2] + BlendshapeCoeff[k] * pt_TB[k-1][2];
		}
		
		pt[0] = pt[0] + Pose[12];
		pt[1] = pt[1] + Pose[13];
		pt[2] = pt[2] + Pose[14];
				
		p_indx[0] = min(m_col-1, max(0, convert_int(round((pt[0]/fabs(pt[2]))*calib[0] + calib[2])))); 
		p_indx[1] = n_row - 1 - min(n_row-1, max(0, convert_int(round((pt[1]/fabs(pt[2]))*calib[1] + calib[3])))); 
		
		int size = 2;
		int li = max(p_indx[1] - size, 0);
		int ui = min(p_indx[1] + size+1, n_row);
		int lj = max(p_indx[0] - size, 0);
		int uj = min(p_indx[0] + size+1, m_col);
		float dist;
		float min_dist = 1000.0f;
		float4 best_n;
		float4 best_v;
		
		for (int i = li; i < ui; i++) {
			for (int j = lj; j < uj; j++) {
			
				nprev = read_imagef(NMap, smp, (int2){j, i});
				
				if (nprev.x == 0.0 && nprev.y == 0.0 && nprev.z == 0.0)
					continue;
				
				vprev = read_imagef(VMap, smp, (int2){j, i}); // (x,y,z, flag)

				dist = sqrt((vprev.x-pt[0])*(vprev.x-pt[0]) + (vprev.y-pt[1])*(vprev.y-pt[1]) + (vprev.z-pt[2])*(vprev.z-pt[2]));
				float dist_angle = nmleBump[0] * nprev.x + nmleBump[1] * nprev.y + nmleBump[2] * nprev.z;

				if (dist < min_dist && dist_angle > angleThres) {
					min_dist = dist;
					best_n.x = nprev.x; best_n.y = nprev.y; best_n.z = nprev.z;
					best_v.x = vprev.x; best_v.y = vprev.y; best_v.z = vprev.z;
				}
			}
		}
		
		if (min_dist > distThres)
			return false;
			
		//printf("pt: %f %f %f \n", pt[0], pt[1], pt[2]);
		//printf("best_v: %f %f %f \n", best_v.x, best_v.y, best_v.z);
			
		for (int k = 0; k < NB_BS-1; k++) {			
			tmpX[k] = (best_n.x*pt_TB[k][0] + best_n.y*pt_TB[k][1] + best_n.z*pt_TB[k][2]);
		}
									
		tmpX[NB_BS-1] = -(best_n.x * (pt[0] - best_v.x) + best_n.y * (pt[1] - best_v.y) + best_n.z * (pt[2] - best_v.z));				
									
		//tmpX[NB_BS-1] = -w*(pt[0] - best_v.x);
		//tmpY[NB_BS-1] = -w*(pt[1] - best_v.y);
		//tmpZ[NB_BS-1] = -w*(pt[2] - best_v.z);							
				
		return true;

}

//LabelsMask contains 4  hannel unsigned char value. the last channel defines pixels of the front face
__kernel void SystemPRbKernel(__read_only image2d_t VMap, __read_only image2d_t NMap, __read_only image2d_t NMapBump, 
						__global float *VerticesBS, __read_only image2d_t Bump, __global float *buf,
						int fact, __constant float *Pose, __constant float *calib, __constant float *BlendshapeCoeff,
						__constant int *LandMarksBump, __constant float *LandMarks,
						__read_only image2d_t LabelsMask, __global int* nbMatches,
						int n_row, int m_col, int n_bump /*height*/, int m_bump /*width*/) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = (i*fact) * m_bump + (j*fact);
	int indx_buff;
	int indx_buffY;
	int indx_buffZ;

    bool found_coresp = false;
	float weight = 1.0;
	
	float row[NB_BS];
	float rowY[NB_BS];
	float rowZ[NB_BS];
	
    if ((i*fact) < n_bump && (j*fact) < m_bump)
        found_coresp = search (tid, (int2){j*fact,i*fact}, calib, Pose, VMap, NMap, NMapBump, VerticesBS, Bump, BlendshapeCoeff, LabelsMask, n_row, m_col, n_bump, m_bump, row, rowY, rowZ);
	
	if (i*fact == n_bump && j < 43) {
		found_coresp = searchLandMark(j, calib, Pose, VMap, VerticesBS, Bump, BlendshapeCoeff, LandMarksBump, LandMarks, n_row, m_col, n_bump, m_bump, row, rowY, rowZ);
	}
	
	if (i*fact == n_bump && j >= 43 && j < 70) {
		found_coresp = true;
		for (int k = 0; k < NB_BS; k++) {
			row[k] = 0.0f;
		}
		row[j-43] = 0.1f;
	}
	
	if (found_coresp) {
		indx_buff = atomic_inc(nbMatches);
		if ((i*fact) < n_bump && (j*fact) < m_bump) {
			//indx_buffY = atomic_inc(nbMatches);
			//indx_buffZ = atomic_inc(nbMatches);
			//#pragma unroll
			for (int k = 0; k < NB_BS; k++) {
				buf[NB_BS*indx_buff + k] = row[k];
				//buf[NB_BS*indx_buffY + k] = rowY[k];
				//buf[NB_BS*indx_buffZ + k] = rowZ[k];
			}
		} else {
			if (i*fact == n_bump && j < 43) {
				indx_buffY = atomic_inc(nbMatches);
				indx_buffZ = atomic_inc(nbMatches);
				#pragma unroll
				for (int k = 0; k < NB_BS; k++) {
					buf[NB_BS*indx_buff + k] = row[k];
					buf[NB_BS*indx_buffY + k] = rowY[k];
					buf[NB_BS*indx_buffZ + k] = rowZ[k];
				}
			} else {
				if (i*fact == n_bump && j < 70) {
					//#pragma unroll
					for (int k = 0; k < NB_BS; k++) {
						buf[NB_BS*indx_buff + k] = row[k];
					}
				}
			}
		}
	}
}