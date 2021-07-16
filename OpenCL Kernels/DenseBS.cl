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

__kernel void DenseBSKernel(__global float *Bump, __global float *Mask, __global Point3DGPU *VerticesBS, __global Point3DGPU *Vertices,
						__global float *WeightMap, __global FaceGPU *Triangles, __global unsigned short *Label, __global float *Pose, __global double *buf, int n_bump /*height*/, int m_bump /*width*/, int BLOCK_SIZE_X, int BLOCK_SIZE_Y) {

	unsigned int i = get_global_id(0); /*height*/
	unsigned int j = get_global_id(1); /*width*/
	unsigned int tid = i*m_bump + j;

	bool valid = true;
		
	if (Label[tid] == 0) 
		valid = false;
		
	FaceGPU CurrFace = Triangles[Label[tid]];
	float weights[3];
	weights[0] = WeightMap[3 * tid];
	weights[1] = WeightMap[3 * tid + 1];
	weights[2] = WeightMap[3 * tid + 2];
	if (weights[0] == -1.0f) 
		valid = false;
		
	float fact_BP, d, tmp ;
	float nmle[3];
	float pt[3];
	float ptRef[3];
	float nmleRef[3];
	float nTmp[3];
	float pTmp[3];
	Point3DGPU V1;
	if (valid) {
		fact_BP = 1000.0f;
		d = Bump[tid] / fact_BP;
		
		nmle[0] = 0.0; nmle[1] = 0.0; nmle[2] = 0.0;
		pt[0] = 0.0; pt[1] = 0.0; pt[2] = 0.0;
		
		pt[0] = (weights[0] * Vertices[CurrFace._v1]._x + weights[1] * Vertices[CurrFace._v2]._x + weights[2] * Vertices[CurrFace._v3]._x) / (weights[0] + weights[1] + weights[2]);
		pt[1] = (weights[0] * Vertices[CurrFace._v1]._y + weights[1] * Vertices[CurrFace._v2]._y + weights[2] * Vertices[CurrFace._v3]._y) / (weights[0] + weights[1] + weights[2]);
		pt[2] = (weights[0] * Vertices[CurrFace._v1]._z + weights[1] * Vertices[CurrFace._v2]._z + weights[2] * Vertices[CurrFace._v3]._z) / (weights[0] + weights[1] + weights[2]);
		
		ptRef[0] = pt[0]; ptRef[1] = pt[1]; ptRef[2] = pt[2];

		nmle[0] = (weights[0] * Vertices[CurrFace._v1]._Nx + weights[1] * Vertices[CurrFace._v2]._Nx + weights[2] * Vertices[CurrFace._v3]._Nx) / (weights[0] + weights[1] + weights[2]);
		nmle[1] = (weights[0] * Vertices[CurrFace._v1]._Ny + weights[1] * Vertices[CurrFace._v2]._Ny + weights[2] * Vertices[CurrFace._v3]._Ny) / (weights[0] + weights[1] + weights[2]);
		nmle[2] = (weights[0] * Vertices[CurrFace._v1]._Nz + weights[1] * Vertices[CurrFace._v2]._Nz + weights[2] * Vertices[CurrFace._v3]._Nz) / (weights[0] + weights[1] + weights[2]);
		tmp = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
		nmle[0] = nmle[0]/tmp;
		nmle[1] = nmle[1]/tmp;
		nmle[2] = nmle[2]/tmp;
		
		//VerticesBS[tid]._x = pt[0]; VerticesBS[tid]._y = pt[1]; VerticesBS[tid]._z = pt[2];
		//VerticesBS[tid]._Nx = nmle[0]; VerticesBS[tid]._Ny = nmle[1]; VerticesBS[tid]._Nz = nmle[2];
		
		VerticesBS[tid]._Tx = pt[0] + d*nmle[0];
		VerticesBS[tid]._Ty = pt[1] + d*nmle[1];
		VerticesBS[tid]._Tz = pt[2] + d*nmle[2]; 
		
		V1 = VerticesBS[tid];
		pt[0] = V1._Tx * Pose[0] + V1._Ty * Pose[4] + V1._Tz * Pose[8];
		pt[1] = V1._Tx * Pose[1] + V1._Ty * Pose[5] + V1._Tz * Pose[9];
		pt[2] = V1._Tx * Pose[2] + V1._Ty * Pose[6] + V1._Tz * Pose[10];
		
		VerticesBS[tid]._Tx = pt[0];
		VerticesBS[tid]._Ty = pt[1];
		VerticesBS[tid]._Tz = pt[2]; 
		
		nmleRef[0] = nmle[0]; nmleRef[1] = nmle[1]; nmleRef[2] = nmle[2];
		
		for (int k = 1; k < 49; k++) {
			nTmp[0] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Nx + weights[1] * Vertices[k*7366 + CurrFace._v2]._Nx + weights[2] * Vertices[k*7366 + CurrFace._v3]._Nx) / (weights[0] + weights[1] + weights[2]);
			nTmp[1] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Ny + weights[1] * Vertices[k*7366 + CurrFace._v2]._Ny + weights[2] * Vertices[k*7366 + CurrFace._v3]._Ny) / (weights[0] + weights[1] + weights[2]);
			nTmp[2] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Nz + weights[1] * Vertices[k*7366 + CurrFace._v2]._Nz + weights[2] * Vertices[k*7366 + CurrFace._v3]._Nz) / (weights[0] + weights[1] + weights[2]);
			tmp = sqrt(nTmp[0]*nTmp[0] + nTmp[1]*nTmp[1] + nTmp[2]*nTmp[2]);
			nTmp[0] = nTmp[0]/tmp;
			nTmp[1] = nTmp[1]/tmp;
			nTmp[2] = nTmp[2]/tmp;

			pTmp[0] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._x + weights[1] * Vertices[k*7366 + CurrFace._v2]._x + weights[2] * Vertices[k*7366 + CurrFace._v3]._x) / (weights[0] + weights[1] + weights[2]);
			pTmp[1] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._y + weights[1] * Vertices[k*7366 + CurrFace._v2]._y + weights[2] * Vertices[k*7366 + CurrFace._v3]._y) / (weights[0] + weights[1] + weights[2]);
			pTmp[2] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._z + weights[1] * Vertices[k*7366 + CurrFace._v2]._z + weights[2] * Vertices[k*7366 + CurrFace._v3]._z) / (weights[0] + weights[1] + weights[2]);
		
			//VerticesBS[k*n_bump*m_bump + tid]._x = pt[0]; VerticesBS[k*n_bump*m_bump +tid]._y = pt[1]; VerticesBS[k*n_bump*m_bump +tid]._z = pt[2];
			//VerticesBS[k*n_bump*m_bump +tid]._Nx = nmle[0]; VerticesBS[k*n_bump*m_bump +tid]._Ny = nmle[1]; VerticesBS[k*n_bump*m_bump + tid]._Nz = nmle[2];
			
			VerticesBS[k*n_bump*m_bump + tid]._Tx = pTmp[0] + d*nTmp[0];
			VerticesBS[k*n_bump*m_bump + tid]._Ty = pTmp[1] + d*nTmp[1];
			VerticesBS[k*n_bump*m_bump + tid]._Tz = pTmp[2] + d*nTmp[2]; 
			
			V1 = VerticesBS[k*n_bump*m_bump + tid];
			pTmp[0] = V1._Tx * Pose[0] + V1._Ty * Pose[4] + V1._Tz * Pose[8];
			pTmp[1] = V1._Tx * Pose[1] + V1._Ty * Pose[5] + V1._Tz * Pose[9];
			pTmp[2] = V1._Tx * Pose[2] + V1._Ty * Pose[6] + V1._Tz * Pose[10];
			
			VerticesBS[k*n_bump*m_bump + tid]._Tx = pTmp[0];
			VerticesBS[k*n_bump*m_bump + tid]._Ty = pTmp[1];
			VerticesBS[k*n_bump*m_bump + tid]._Tz = pTmp[2]; 
		}
	}
													  
	////////////// Compute J^t*J ///////////////////////////
	
	__local double smem[THREAD_SIZE*THREAD_SIZE];
	
	int tid_b = get_local_id(1)*THREAD_SIZE + get_local_id(0);
	
    int shift = 0;
	#pragma unroll
	for (int k = 1; k < 49; k++) {
		for (int l = k; l < 49; l++) {
			if (valid) {
				smem[tid_b] = (double) ((VerticesBS[k*n_bump*m_bump + tid]._Tx - pt[0])*(VerticesBS[l*n_bump*m_bump + tid]._Tx  - pt[0])
							+ (VerticesBS[k*n_bump*m_bump + tid]._Ty - pt[1])*(VerticesBS[l*n_bump*m_bump + tid]._Ty  - pt[1])
							+ (VerticesBS[k*n_bump*m_bump + tid]._Tz - pt[2])*(VerticesBS[l*n_bump*m_bump + tid]._Tz  - pt[2]));
							//printf("smem[tid_b]: %f %f %f %f %f %f\n", VerticesBS[k*n_bump*m_bump + tid]._Tx,  pt[0], VerticesBS[k*n_bump*m_bump + tid]._Ty,  pt[1], VerticesBS[k*n_bump*m_bump + tid]._Tz,  pt[2]);
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