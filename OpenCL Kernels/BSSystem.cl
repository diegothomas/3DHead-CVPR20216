#pragma OPENCL EXTENSION cl_amd_printf : enable

#define THREAD_SIZE 16

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

typedef struct tag_FaceGPU {
	int _v1;
	int _v2;
	int _v3;
} FaceGPU;

typedef struct tag_Point3DGPU {
	//Position
	float _x;
	float _y;
	float _z;

	// Transformed Position when deformation is happening
	float _Tx;
	float _Ty;
	float _Tz;

	//Normal vector
	float _Nx;
	float _Ny;
	float _Nz;

	// Transformed Normal vector when deformation is happening
	float _TNx;
	float _TNy;
	float _TNz;

	//Color vector
	float _R;
	float _G;
	float _B;

	//texture coordinates
	float _u;
	float _v;

	unsigned char _flag;

	// Index
	int _indx;
	bool _BackPoint;
	
} Point3DGPU;

inline int Myround(float a) {
	int res = (int)a;
	if (a - ((float)res) > 0.5f)
		return res + 1;
	else
		return res;
}

__kernel void BSSystemKernel(__global double * buf, __global float *VMap, __global float *NMap, __global Point3DGPU *VerticesBS, __global float *Mask, __global float *BlendshapeCoeff, 
							__global float *Pose, __global float *intrinsic, int n, int m, int n_bump /*height*/, int m_bump /*width*/, int BLOCK_SIZE_X, int BLOCK_SIZE_Y) {

	unsigned int i = get_global_id(0); /*height*/
	unsigned int j = get_global_id(1); /*width*/
	unsigned int tid = i*m_bump + j;
		
	bool valid = true;
	if (Mask[tid] == 0.0f) 
		valid = false;
	
	int p_indx[2];
	float pt[3];	
	float ptRef[3];
	float min_dist = 1000000.0f;
	float tmpPt[3];
	float tmpNMLE[3];
	float pointClose[3];
	float nmleClose[3];
	float dist;
	int li, ui, lj, uj;
	
	if (valid) {
		pt[0] = VerticesBS[tid]._Tx;
		pt[1] = VerticesBS[tid]._Ty;
		pt[2] = VerticesBS[tid]._Tz;
		ptRef[0] = pt[0]; ptRef[1] = pt[1]; ptRef[2] = pt[2]; 

		for (int k = 1; k < 49; k++) {
			pt[0] = pt[0] + BlendshapeCoeff[k] * (VerticesBS[k*n_bump*m_bump + tid]._Tx - ptRef[0]);
			pt[1] = pt[1] + BlendshapeCoeff[k] * (VerticesBS[k*n_bump*m_bump + tid]._Ty - ptRef[1]);
			pt[2] = pt[2] + BlendshapeCoeff[k] * (VerticesBS[k*n_bump*m_bump + tid]._Tz - ptRef[2]);
		}
		pt[0] = pt[0] + Pose[12];
		pt[1] = pt[1] + Pose[13];
		pt[2] = pt[2] + Pose[14];

		// 2. Compute associated points
		// Project the point onto the depth image
		p_indx[0] = min(m - 1, max(0, Myround((pt[0] / fabs(pt[2]))*intrinsic[0] + intrinsic[2])));
		p_indx[1] = min(n - 1, max(0, Myround((pt[1] / fabs(pt[2]))*intrinsic[1] + intrinsic[3])));
		
		li = max(n - 1 - p_indx[1] - 1, 0);
		ui = min(n - 1 - p_indx[1] + 2, n);
		lj = max(p_indx[0] - 1, 0);
		uj = min(p_indx[0] + 2, m);
		for (int indx_i = li; indx_i < ui; indx_i++) {
			for (int indx_j = lj; indx_j < uj; indx_j++) {
				tmpPt[0] = VMap[3*(indx_i*m + indx_j)];
				tmpPt[1] = VMap[3*(indx_i*m + indx_j) + 1];
				tmpPt[2] = VMap[3*(indx_i*m + indx_j) + 2];

				
				tmpNMLE[0] = NMap[3*(indx_i*m + indx_j)];
				tmpNMLE[1] = NMap[3*(indx_i*m + indx_j) + 1];
				tmpNMLE[2] = NMap[3*(indx_i*m + indx_j) + 2];
				if (tmpNMLE[0] == 0.0 && tmpNMLE[1] == 0.0 && tmpNMLE[2] == 0.0f)
					continue;

				dist = sqrt((tmpPt[0] - pt[0])*(tmpPt[0] - pt[0]) + (tmpPt[1] - pt[1])*(tmpPt[1] - pt[1]) + (tmpPt[2] - pt[2])*(tmpPt[2] - pt[2]));

				if (dist < min_dist) {
					min_dist = dist;
					pointClose[0] = tmpPt[0];
					pointClose[1] = tmpPt[1];
					pointClose[2] = tmpPt[2];
					nmleClose[0] = tmpNMLE[0];
					nmleClose[1] = tmpNMLE[1];
					nmleClose[2] = tmpNMLE[2];
				}
			}
		}
	}
	
	if (min_dist > 0.02f) {
		valid = false;
	}
	
	////////////// Compute J^t*b ///////////////////////////
	
	__local double smem[THREAD_SIZE*THREAD_SIZE];
	
	int tid_b = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	
    int shift = 0;
	#pragma unroll
	for (int k = 1; k < 49; k++) {
		if (valid) {
			smem[tid_b] = (double) ((VerticesBS[k*n_bump*m_bump + tid]._Tx - ptRef[0])*(pointClose[0] - pt[0])
						+ (VerticesBS[k*n_bump*m_bump + tid]._Ty - ptRef[1])*(pointClose[1] - pt[1])
						+ (VerticesBS[k*n_bump*m_bump + tid]._Tz - ptRef[2])*(pointClose[2] - pt[2]));
		} else {
			smem[tid_b] = 0.0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		reduce(smem, THREAD_SIZE*THREAD_SIZE);
		
		if (tid_b == 0) {			
			buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
		}
	}
	
	// compute current residual
	if (valid) {
		smem[tid_b] = (double)min_dist;
	} else {
		smem[tid_b] = 0.0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid_b == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
	
	// compute number of matches
	if (valid) {
		smem[tid_b] = 1.0;
	} else {
		smem[tid_b] = 0.0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(smem, THREAD_SIZE*THREAD_SIZE);
	
	if (tid_b == 0) {	
		buf[get_group_id(0) + get_group_id(1)*BLOCK_SIZE_X + (shift++)*(BLOCK_SIZE_X*BLOCK_SIZE_Y)] = smem[0];
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//float scal = (_Vtx[k][3 * i] - PtRef[0])*nmleClose[0] + (_Vtx[k][3 * i + 1] - PtRef[1])*nmleClose[1] + (_Vtx[k][3 * i + 2] - PtRef[2])*nmleClose[2];
	//float scal2 = (pointClose[0] - pt[3 * i])*nmleClose[0] + (pointClose[1] - pt[3 * i + 1])*nmleClose[1] + (pointClose[2] - pt[3 * i + 2])*nmleClose[2];
	//b(k - 1) = b(k - 1) + (scal * scal2);

	/*b(k - 1) = b(k - 1) + (_Vtx[k][3 * i] - PtRef[0])*(pointClose[0] - pt[3 * i])
		+ (_Vtx[k][3 * i + 1] - PtRef[1])*(pointClose[1] - pt[3 * i + 1])
		+ (_Vtx[k][3 * i + 2] - PtRef[2])*(pointClose[2] - pt[3 * i + 2]);*/

	/*for (int l = 1; l < 49; l++) {
		scal = (_Vtx[k][3 * i] - PtRef[0])*nmleClose[0] + (_Vtx[k][3 * i + 1] - PtRef[1])*nmleClose[1] + (_Vtx[k][3 * i + 2] - PtRef[2])*nmleClose[2];
		scal2 = (_Vtx[l][3 * i] - PtRef[0])*nmleClose[0] + (_Vtx[l][3 * i + 1] - PtRef[1])*nmleClose[1] + (_Vtx[l][3 * i + 2] - PtRef[2])*nmleClose[2];
		JTJ(k - 1, l - 1) = JTJ(k - 1, l - 1) + (scal * scal2);
	}*/
}