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

__kernel void BumpKernel(__global float *VMap, __global float *NMap, __global unsigned char *RGBMap, __global unsigned char *Flag, __global float *Bump, __global float *Mask,
						__global Point3DGPU *VerticesBump, __global float *BlendshapeCoeff, __global Point3DGPU *Vertices, __global unsigned char *LabelsMask, 
						__global float *WeightMap, __global FaceGPU *Triangles, __global unsigned short *Label, __global float *Pose, __global float *calib, 
						int n, int m, int n_bump /*height*/, int m_bump /*width*/) {

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
		
	unsigned char flag = 0;
	if ((LabelsMask[3 * tid + 2] == 255) && (LabelsMask[3 * tid + 1] == 0) && (LabelsMask[3 * tid] == 0))
		flag = 1; // Left eye up
	if ((LabelsMask[3 * tid + 2] == 0) && (LabelsMask[3 * tid + 1] == 255) && (LabelsMask[3 * tid] == 0))
		flag = 2; // Left eye down
	if ((LabelsMask[3 * tid + 2] == 0) && (LabelsMask[3 * tid + 1] == 0) && (LabelsMask[3 * tid] == 255))
		flag = 3; // right eye up
	if ((LabelsMask[3 * tid + 2] == 255) && (LabelsMask[3 * tid + 1] == 0) && (LabelsMask[3 * tid] == 255))
		flag = 4; // right eye down
	if ((LabelsMask[3 * tid + 2] == 255) && (LabelsMask[3 * tid + 1] == 255) && (LabelsMask[3 * tid] == 0))
		flag = 5; // mouth up
	if ((LabelsMask[3 * tid + 2] == 0) && (LabelsMask[3 * tid + 1] == 255) && (LabelsMask[3 * tid] == 255))
		flag = 6; // mouth down
		
	float nmle[3];
	nmle[0] = 0.0; nmle[1] = 0.0; nmle[2] = 0.0;
	float pt[3];
	pt[0] = 0.0; pt[1] = 0.0; pt[2] = 0.0;
	
	float fact_BP = 1000.0f;
	float d = (Mask[tid] > 0.0f) ? Bump[tid] / fact_BP : 0.0f;
	
	pt[0] = (weights[0] * Vertices[CurrFace._v1]._x + weights[1] * Vertices[CurrFace._v2]._x + weights[2] * Vertices[CurrFace._v3]._x) / (weights[0] + weights[1] + weights[2]);
	pt[1] = (weights[0] * Vertices[CurrFace._v1]._y + weights[1] * Vertices[CurrFace._v2]._y + weights[2] * Vertices[CurrFace._v3]._y) / (weights[0] + weights[1] + weights[2]);
	pt[2] = (weights[0] * Vertices[CurrFace._v1]._z + weights[1] * Vertices[CurrFace._v2]._z + weights[2] * Vertices[CurrFace._v3]._z) / (weights[0] + weights[1] + weights[2]);

	nmle[0] = (weights[0] * Vertices[CurrFace._v1]._Nx + weights[1] * Vertices[CurrFace._v2]._Nx + weights[2] * Vertices[CurrFace._v3]._Nx) / (weights[0] + weights[1] + weights[2]);
	nmle[1] = (weights[0] * Vertices[CurrFace._v1]._Ny + weights[1] * Vertices[CurrFace._v2]._Ny + weights[2] * Vertices[CurrFace._v3]._Ny) / (weights[0] + weights[1] + weights[2]);
	nmle[2] = (weights[0] * Vertices[CurrFace._v1]._Nz + weights[1] * Vertices[CurrFace._v2]._Nz + weights[2] * Vertices[CurrFace._v3]._Nz) / (weights[0] + weights[1] + weights[2]);
	float tmp = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
	nmle[0] = nmle[0]/tmp;
	nmle[1] = nmle[1]/tmp;
	nmle[2] = nmle[2]/tmp;
	
	pt[0] = pt[0] + d*nmle[0]; 
	pt[1] = pt[1] + d*nmle[1];  
	pt[2] = pt[2] + d*nmle[2]; 
	float ptRef[3];
	ptRef[0] = pt[0]; ptRef[1] = pt[1]; ptRef[2] = pt[2];
	
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
	
		pTmp[0] = pTmp[0] + d*nTmp[0]; 
		pTmp[1] = pTmp[1] + d*nTmp[1];  
		pTmp[2] = pTmp[2] + d*nTmp[2]; 
	
		// This blended normal is not really a normal since it may be not normalized
		nmle[0] = nmle[0] + (nTmp[0] - nmleRef[0]) * BlendshapeCoeff[k];
		nmle[1] = nmle[1] + (nTmp[1] - nmleRef[1]) * BlendshapeCoeff[k];
		nmle[2] = nmle[2] + (nTmp[2] - nmleRef[2]) * BlendshapeCoeff[k];

		pt[0] = pt[0] + (pTmp[0] - ptRef[0]) * BlendshapeCoeff[k];
		pt[1] = pt[1] + (pTmp[1] - ptRef[1]) * BlendshapeCoeff[k];
		pt[2] = pt[2] + (pTmp[2] - ptRef[2]) * BlendshapeCoeff[k];
	}

	int p_indx[2];
	
	VerticesBump[tid]._x = pt[0]; VerticesBump[tid]._y = pt[1]; VerticesBump[tid]._z = pt[2];
	VerticesBump[tid]._Nx = nmle[0]; VerticesBump[tid]._Ny = nmle[1]; VerticesBump[tid]._Nz = nmle[2];
	
	Point3DGPU V1 = VerticesBump[tid];
	pt[0] = V1._x * Pose[0] + V1._y * Pose[4] + V1._z * Pose[8] + Pose[12];
	pt[1] = V1._x * Pose[1] + V1._y * Pose[5] + V1._z * Pose[9] + Pose[13];
	pt[2] = V1._x * Pose[2] + V1._y * Pose[6] + V1._z * Pose[10] + Pose[14];
	
	nmle[0] = V1._Nx * Pose[0] + V1._Ny * Pose[4] + V1._Nz * Pose[8];
	nmle[1] = V1._Nx * Pose[1] + V1._Ny * Pose[5] + V1._Nz * Pose[9];
	nmle[2] = V1._Nx * Pose[2] + V1._Ny * Pose[6] + V1._Nz * Pose[10];
	
	if (nmle[2] < 0.0f) {
		if (Mask[tid] > 0.0f) {
			VerticesBump[tid]._Tx = V1._x;
			VerticesBump[tid]._Ty = V1._y;
			VerticesBump[tid]._Tz = V1._z; 
		}
		return;
	}
	
	float Tnmle[3];
	Tnmle[0] = V1._TNx * Pose[0] + V1._TNy * Pose[4] + V1._TNz * Pose[8];
	Tnmle[1] = V1._TNx * Pose[1] + V1._TNy * Pose[5] + V1._TNz * Pose[9];
	Tnmle[2] = V1._TNx * Pose[2] + V1._TNy * Pose[6] + V1._TNz * Pose[10];
	if (Tnmle[0] == 0.0f && Tnmle[1] == 0.0f && Tnmle[2] == 0.0f) {
		Tnmle[0] = nmle[0];
		Tnmle[1] = nmle[1];
		Tnmle[2] = nmle[2];
	}
	
	float bum_val = d; //Bump[tid];
	float min_dist = 1000000000.0f;
	float best_state = -1.0f;
	float fact_curr = Myround(Mask[tid]) == 0 ? 1.0 : min(2.0f, sqrt((float)Myround(Mask[tid])));
	float pos[3];
	float ptIn[3];
	
	//summit 1
	d = (bum_val - (float)(MESSAGE_LENGTH / 2) / fact_curr) / fact_BP;
	pos[0] = pt[0] + d*nmle[0];
	pos[1] = pt[1] + d*nmle[1];
	pos[2] = pt[2] + d*nmle[2];
	// Project the point onto the depth image
	int s1[2];
	s1[0] = min(m_bump - 1, max(0, Myround((pos[0] / fabs(pos[2]))*calib[0] + calib[2])));
	s1[1] = min(n_bump - 1, max(0, Myround((pos[1] / fabs(pos[2]))*calib[1] + calib[3])));
	
	//summit 2
	d = (bum_val + (float)(MESSAGE_LENGTH / 2) / fact_curr) / fact_BP;
	pos[0] = pt[0] + d*nmle[0];
	pos[1] = pt[1] + d*nmle[1];
	pos[2] = pt[2] + d*nmle[2];
	// Project the point onto the depth image
	int s2[2];
	s2[0] = min(m_bump - 1, max(0, Myround((pos[0] / fabs(pos[2]))*calib[0] + calib[2])));
	s2[1] = min(n_bump - 1, max(0, Myround((pos[1] / fabs(pos[2]))*calib[1] + calib[3])));
	
	float length = sqrt((float)((s1[0]-s2[0])*(s1[0]-s2[0])) + (float)((s1[1]-s2[1])*(s1[1]-s2[1])));
	
	float dir[2];
	dir[0] = ((float)(s2[0]-s1[0]))/length;
	dir[1] = ((float)(s2[1]-s1[1]))/length;
	
	d = bum_val / fact_BP;
	pos[0] = pt[0] + d*nmle[0];
	pos[1] = pt[1] + d*nmle[1];
	pos[2] = pt[2] + d*nmle[2];
	
	for (float lambda = 0.0f; lambda <= length; lambda += 0.5f) {
		int k = n-1-Myround((float)s1[1] + lambda*dir[1]);
		int l = Myround((float)s1[0] + lambda*dir[0]);
		
		if (k < 0 || k > n - 1 || l < 0 || l > m - 1)
			continue;

		ptIn[0] = VMap[3*(k*m + l)];
		ptIn[1] = VMap[3*(k*m + l) + 1];
		ptIn[2] = VMap[3*(k*m + l) + 2];
		
		if (ptIn[0] == 0.0f && ptIn[1] == 0.0f && ptIn[2] == 0.0f)
			continue;

		//compute distance of point to the normal
		float u_vect[3];
		u_vect[0] = ptIn[0] - pt[0];
		u_vect[1] = ptIn[1] - pt[1];
		u_vect[2] = ptIn[2] - pt[2];

		float proj = u_vect[0] * nmle[0] + u_vect[1] * nmle[1] + u_vect[2] * nmle[2];
		float v_vect[3];
		v_vect[0] = u_vect[0] - proj * nmle[0];
		v_vect[1] = u_vect[1] - proj * nmle[1];
		v_vect[2] = u_vect[2] - proj * nmle[2];
		float dist = sqrt((ptIn[0] - pos[0]) * (ptIn[0] - pos[0]) + (ptIn[1]- pos[1]) * (ptIn[1] - pos[1]) + (ptIn[2] - pos[2]) * (ptIn[2] - pos[2]));
		float dist_to_nmle = sqrt(v_vect[0] * v_vect[0] + v_vect[1] * v_vect[1] + v_vect[2] * v_vect[2]);
		float dist_angle = Tnmle[0] * NMap[3*(k*m + l)] + Tnmle[1] * NMap[3*(k*m + l) + 1] + Tnmle[2] * NMap[3*(k*m + l) + 2];
		bool valid = (flag == 0) || (flag == Flag[k*m + l]);

		if (dist_to_nmle < min_dist && dist_angle > 0.6 && valid && dist < 0.04) {
			min_dist = dist_to_nmle;
			best_state = proj * fact_BP;
		}
 	}
	
	if (best_state == -1.0 || min_dist > 0.005) {
		if (Mask[tid] > 0.0) {
			d = Bump[tid] / fact_BP;
			VerticesBump[tid]._Tx = V1._x + d*V1._Nx;
			VerticesBump[tid]._Ty = V1._y + d*V1._Ny;
			VerticesBump[tid]._Tz = V1._z + d*V1._Nz; 
		}
		return;
	}

	float weight = Mask[tid] == 0.0 ? 0.1 : Tnmle[2];
	weight = weight*weight;
	float new_bump = (weight*best_state + bum_val*Mask[tid]) / (Mask[tid] + weight);
	if (Mask[tid] < 100.0) {
		Bump[tid] = new_bump;
	}

	//Get color
	float p1[3];
	d = Bump[tid] / fact_BP;
	p1[0] = pt[0] + d*nmle[0];
	p1[1] = pt[1] + d*nmle[1];
	p1[2] = pt[2] + d*nmle[2];

	p_indx[0] = min(m - 1, max(0, Myround((p1[0] / fabs(p1[2]))*calib[0] + calib[2])));
	p_indx[1] = n - 1 - min(n - 1, max(0, Myround((p1[1] / fabs(p1[2]))*calib[1] + calib[3])));
	if ((VMap[3*(p_indx[1] * m + p_indx[0])] != 0.0f || VMap[3*(p_indx[1] * m + p_indx[0]) + 1] != 0.0f || VMap[3*(p_indx[1] * m + p_indx[0]) + 2] != 0.0f) && Mask[tid] < 100.0f) {
		VerticesBump[tid]._R = (weight*((float)RGBMap[3*(p_indx[1] * m + p_indx[0]) + 2]) + V1._R*Mask[tid]) / (Mask[tid] + weight);  
		VerticesBump[tid]._G = (weight*((float)RGBMap[3*(p_indx[1] * m + p_indx[0]) + 1]) + V1._G*Mask[tid]) / (Mask[tid] + weight);
		VerticesBump[tid]._B = (weight*((float)RGBMap[3*(p_indx[1] * m + p_indx[0])]) + V1._B*Mask[tid]) / (Mask[tid] + weight); 
	}

	if (Mask[tid] < 100.0) {
		Mask[tid] = Mask[tid] + weight;
	}

	if (Mask[tid] > 0.0) {
		d = Bump[tid] / fact_BP;
		VerticesBump[tid]._Tx = V1._x + d*V1._Nx;
		VerticesBump[tid]._Ty = V1._y + d*V1._Ny;
		VerticesBump[tid]._Tz = V1._z + d*V1._Nz; 
	}
	
}