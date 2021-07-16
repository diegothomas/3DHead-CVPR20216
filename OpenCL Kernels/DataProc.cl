#pragma OPENCL EXTENSION cl_amd_printf : enable

#define	NB_BS 28

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

__kernel void DataProcKernel(__global float *VerticesBS,
						__read_only image2d_t Bump, 
						__global Point3DGPU *Vertices, 
						__read_only image2d_t WeightMap, 
						__global FaceGPU *Triangles, 
						int n_bump /*height*/, int m_bump /*width*/) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i*m_bump + j;
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	if (i > n_bump-1 || j > m_bump-1)
		return;
		
	float4 bumpIn = read_imagef(Bump, smp, (int2) {j,i}); // (bump, mask, label, 0)
	float4 weights = read_imagef(WeightMap, smp, (int2) {j,i});
	
	if (bumpIn.z == -1.0f) {
		for (int k = 0; k < NB_BS; k++) {
			VerticesBS[k*6*n_bump*m_bump + 6*tid] = 0.0f;
			VerticesBS[k*6*n_bump*m_bump + 6*tid + 1] = 0.0f;
			VerticesBS[k*6*n_bump*m_bump + 6*tid + 2] = 0.0f;
			VerticesBS[k*6*n_bump*m_bump + 6*tid + 3] = 0.0f;
			VerticesBS[k*6*n_bump*m_bump + 6*tid + 4] = 0.0f;
			VerticesBS[k*6*n_bump*m_bump + 6*tid + 5] = 0.0f;
		}
		return;
	}
	FaceGPU CurrFace = Triangles[convert_int(bumpIn.z)];
		
	float nmle[3] = {0.0f,0.0f,0.0f};
	float pt[3] = {0.0f,0.0f,0.0f};
	
	pt[0] = (weights.x * Vertices[CurrFace._v1]._x + weights.y * Vertices[CurrFace._v2]._x + weights.z * Vertices[CurrFace._v3]._x);
	pt[1] = (weights.x * Vertices[CurrFace._v1]._y + weights.y * Vertices[CurrFace._v2]._y + weights.z * Vertices[CurrFace._v3]._y);
	pt[2] = (weights.x * Vertices[CurrFace._v1]._z + weights.y * Vertices[CurrFace._v2]._z + weights.z * Vertices[CurrFace._v3]._z);
	float ptRef[3] = {pt[0],pt[1],pt[2]};

	nmle[0] = (weights.x * Vertices[CurrFace._v1]._Nx + weights.y * Vertices[CurrFace._v2]._Nx + weights.z * Vertices[CurrFace._v3]._Nx);
	nmle[1] = (weights.x * Vertices[CurrFace._v1]._Ny + weights.y * Vertices[CurrFace._v2]._Ny + weights.z * Vertices[CurrFace._v3]._Ny);
	nmle[2] = (weights.x * Vertices[CurrFace._v1]._Nz + weights.y * Vertices[CurrFace._v2]._Nz + weights.z * Vertices[CurrFace._v3]._Nz);
	float tmp = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
	nmle[0] = nmle[0]/tmp;
	nmle[1] = nmle[1]/tmp;
	nmle[2] = nmle[2]/tmp;
	
	float nmleRef[3] = {nmle[0],nmle[1],nmle[2]};
	
	VerticesBS[6*tid] = ptRef[0];
	VerticesBS[6*tid + 1] = ptRef[1];
	VerticesBS[6*tid + 2] = ptRef[2];
	VerticesBS[6*tid + 3] = nmleRef[0];
	VerticesBS[6*tid + 4] = nmleRef[1];
	VerticesBS[6*tid + 5] = nmleRef[2];
	
	float nTmp[3];
	float pTmp[3];
	for (int k = 1; k < NB_BS; k++) {
		nTmp[0] = (weights.x * Vertices[k*4325 + CurrFace._v1]._Nx + weights.y * Vertices[k*4325 + CurrFace._v2]._Nx + weights.z * Vertices[k*4325 + CurrFace._v3]._Nx);
		nTmp[1] = (weights.x * Vertices[k*4325 + CurrFace._v1]._Ny + weights.y * Vertices[k*4325 + CurrFace._v2]._Ny + weights.z * Vertices[k*4325 + CurrFace._v3]._Ny);
		nTmp[2] = (weights.x * Vertices[k*4325 + CurrFace._v1]._Nz + weights.y * Vertices[k*4325 + CurrFace._v2]._Nz + weights.z * Vertices[k*4325 + CurrFace._v3]._Nz);
		float tmp = sqrt(nTmp[0]*nTmp[0] + nTmp[1]*nTmp[1] + nTmp[2]*nTmp[2]);
		nTmp[0] = nTmp[0]/tmp;
		nTmp[1] = nTmp[1]/tmp;
		nTmp[2] = nTmp[2]/tmp;

		pTmp[0] = (weights.x * Vertices[k*4325 + CurrFace._v1]._x + weights.y * Vertices[k*4325 + CurrFace._v2]._x + weights.z * Vertices[k*4325 + CurrFace._v3]._x);
		pTmp[1] = (weights.x * Vertices[k*4325 + CurrFace._v1]._y + weights.y * Vertices[k*4325 + CurrFace._v2]._y + weights.z * Vertices[k*4325 + CurrFace._v3]._y);
		pTmp[2] = (weights.x * Vertices[k*4325 + CurrFace._v1]._z + weights.y * Vertices[k*4325 + CurrFace._v2]._z + weights.z * Vertices[k*4325 + CurrFace._v3]._z);
	
		// This blended normal is not really a normal since it may be not normalized
		VerticesBS[k*6*n_bump*m_bump + 6*tid] = (pTmp[0] - ptRef[0]);
		VerticesBS[k*6*n_bump*m_bump + 6*tid + 1] = (pTmp[1] - ptRef[1]);
		VerticesBS[k*6*n_bump*m_bump + 6*tid + 2] = (pTmp[2] - ptRef[2]);
		VerticesBS[k*6*n_bump*m_bump + 6*tid + 3] = (nTmp[0] - nmleRef[0]);
		VerticesBS[k*6*n_bump*m_bump + 6*tid + 4] = (nTmp[1] - nmleRef[1]);
		VerticesBS[k*6*n_bump*m_bump + 6*tid + 5] = (nTmp[2] - nmleRef[2]);
	}		
}