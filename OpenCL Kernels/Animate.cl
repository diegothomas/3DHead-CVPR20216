#pragma OPENCL EXTENSION cl_amd_printf : enable

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

#define	MESSAGE_LENGTH 40

__kernel void AnimateKernel(__global float *Bump, __global float *Mask, __global Point3DGPU *VerticesBump, __global float *BlendshapeCoeff, __global Point3DGPU *Vertices,
						__global float *WeightMap, __global FaceGPU *Triangles, __global unsigned short *Label, int n_bump /*height*/, int m_bump /*width*/) {

	unsigned int i = get_global_id(0); /*height*/
	unsigned int j = get_global_id(1); /*width*/
	unsigned int tid = i*m_bump + j;

	if (i > n_bump-1 || j > m_bump-1)
		return;
		
	if (Label[tid] == 0) 
		return;
		
	FaceGPU CurrFace = Triangles[Label[tid]];
	float weights[3];
	weights[0] = WeightMap[3 * tid];
	weights[1] = WeightMap[3 * tid + 1];
	weights[2] = WeightMap[3 * tid + 2];
	if (weights[0] == -1.0f) 
		return;
		
	float nmle[3];
	nmle[0] = 0.0; nmle[1] = 0.0; nmle[2] = 0.0;
	float pt[3];
	pt[0] = 0.0; pt[1] = 0.0; pt[2] = 0.0;
	
	pt[0] = (weights[0] * Vertices[CurrFace._v1]._x + weights[1] * Vertices[CurrFace._v2]._x + weights[2] * Vertices[CurrFace._v3]._x) / (weights[0] + weights[1] + weights[2]);
	pt[1] = (weights[0] * Vertices[CurrFace._v1]._y + weights[1] * Vertices[CurrFace._v2]._y + weights[2] * Vertices[CurrFace._v3]._y) / (weights[0] + weights[1] + weights[2]);
	pt[2] = (weights[0] * Vertices[CurrFace._v1]._z + weights[1] * Vertices[CurrFace._v2]._z + weights[2] * Vertices[CurrFace._v3]._z) / (weights[0] + weights[1] + weights[2]);
	float ptRef[3];
	ptRef[0] = pt[0]; ptRef[1] = pt[1]; ptRef[2] = pt[2];

	nmle[0] = (weights[0] * Vertices[CurrFace._v1]._Nx + weights[1] * Vertices[CurrFace._v2]._Nx + weights[2] * Vertices[CurrFace._v3]._Nx) / (weights[0] + weights[1] + weights[2]);
	nmle[1] = (weights[0] * Vertices[CurrFace._v1]._Ny + weights[1] * Vertices[CurrFace._v2]._Ny + weights[2] * Vertices[CurrFace._v3]._Ny) / (weights[0] + weights[1] + weights[2]);
	nmle[2] = (weights[0] * Vertices[CurrFace._v1]._Nz + weights[1] * Vertices[CurrFace._v2]._Nz + weights[2] * Vertices[CurrFace._v3]._Nz) / (weights[0] + weights[1] + weights[2]);
	float tmp = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
	nmle[0] = nmle[0]/tmp;
	nmle[1] = nmle[1]/tmp;
	nmle[2] = nmle[2]/tmp;
	
	float nmleRef[3];
	nmleRef[0] = nmle[0]; nmleRef[1] = nmle[1]; nmleRef[2] = nmle[2];
	
	float nTmp[3];
	float pTmp[3];
	for (int k = 1; k < 49; k++) {
		nTmp[0] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Nx + weights[1] * Vertices[k*7366 + CurrFace._v2]._Nx + weights[2] * Vertices[k*7366 + CurrFace._v3]._Nx) / (weights[0] + weights[1] + weights[2]);
		nTmp[1] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Ny + weights[1] * Vertices[k*7366 + CurrFace._v2]._Ny + weights[2] * Vertices[k*7366 + CurrFace._v3]._Ny) / (weights[0] + weights[1] + weights[2]);
		nTmp[2] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._Nz + weights[1] * Vertices[k*7366 + CurrFace._v2]._Nz + weights[2] * Vertices[k*7366 + CurrFace._v3]._Nz) / (weights[0] + weights[1] + weights[2]);
		float tmp = sqrt(nTmp[0]*nTmp[0] + nTmp[1]*nTmp[1] + nTmp[2]*nTmp[2]);
		nTmp[0] = nTmp[0]/tmp;
		nTmp[1] = nTmp[1]/tmp;
		nTmp[2] = nTmp[2]/tmp;

		pTmp[0] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._x + weights[1] * Vertices[k*7366 + CurrFace._v2]._x + weights[2] * Vertices[k*7366 + CurrFace._v3]._x) / (weights[0] + weights[1] + weights[2]);
		pTmp[1] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._y + weights[1] * Vertices[k*7366 + CurrFace._v2]._y + weights[2] * Vertices[k*7366 + CurrFace._v3]._y) / (weights[0] + weights[1] + weights[2]);
		pTmp[2] = (weights[0] * Vertices[k*7366 + CurrFace._v1]._z + weights[1] * Vertices[k*7366 + CurrFace._v2]._z + weights[2] * Vertices[k*7366 + CurrFace._v3]._z) / (weights[0] + weights[1] + weights[2]);
	
		// This blended normal is not really a normal since it may be not normalized
		nmle[0] = nmle[0] + (nTmp[0] - nmleRef[0]) * BlendshapeCoeff[k];
		nmle[1] = nmle[1] + (nTmp[1] - nmleRef[1]) * BlendshapeCoeff[k];
		nmle[2] = nmle[2] + (nTmp[2] - nmleRef[2]) * BlendshapeCoeff[k];

		pt[0] = pt[0] + (pTmp[0] - ptRef[0]) * BlendshapeCoeff[k];
		pt[1] = pt[1] + (pTmp[1] - ptRef[1]) * BlendshapeCoeff[k];
		pt[2] = pt[2] + (pTmp[2] - ptRef[2]) * BlendshapeCoeff[k];
	}

	VerticesBump[tid]._x = pt[0]; VerticesBump[tid]._y = pt[1]; VerticesBump[tid]._z = pt[2];
	VerticesBump[tid]._Nx = nmle[0]; VerticesBump[tid]._Ny = nmle[1]; VerticesBump[tid]._Nz = nmle[2];
	
	float d;
	float fact_BP = 1000.0f;
	if (Mask[tid] > 0.0f) {
		d = Bump[tid] / fact_BP;
		VerticesBump[tid]._Tx = VerticesBump[tid]._x + d*VerticesBump[tid]._Nx;
		VerticesBump[tid]._Ty = VerticesBump[tid]._y + d*VerticesBump[tid]._Ny;
		VerticesBump[tid]._Tz = VerticesBump[tid]._z + d*VerticesBump[tid]._Nz; 
	}
}