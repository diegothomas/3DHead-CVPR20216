#pragma OPENCL EXTENSION cl_amd_printf : enable

#define	MESSAGE_LENGTH 40
#define	NB_BS 28

//LabelsMask contains 4  hannel unsigned char value. the last channel defines pixels of the front face
__kernel void BumpKernel(__read_only image2d_t VMap, __read_only image2d_t NMap, __read_only image2d_t RGBMap,
						__read_only image2d_t Bump, __write_only image2d_t BumpSwap, 
						__read_only image2d_t RGBMapBump, __write_only image2d_t RGBMapBumpSwap,
						__write_only image2d_t VMapBump, __read_only image2d_t NMapBump,
						__constant float *BlendshapeCoeff, 
						__global float *VerticesBS, 
						__read_only image2d_t LabelsMask, 
						__constant float *Pose, __constant float *calib, 
						int n, int m, int n_bump /*height*/, int m_bump /*width*/, int stable_state) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i*m_bump + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	if (i > n_bump-1 || j > m_bump-1)
		return;
		
	float4 bumpIn = read_imagef(Bump, smp, coords); // (bump, mask, label, 0)
	
	float4 zero = {0.0f,0.0f,0.0f,0.0f};
	if (bumpIn.z == -1.0f) {
		write_imagef(VMapBump, coords, zero);
		write_imagef(BumpSwap, coords, zero);
		write_imagef(RGBMapBumpSwap, coords, zero);
		return;
	}
		
	write_imagef(BumpSwap, coords, bumpIn);
	
	float4 RGBIn = read_imagef(RGBMapBump, smp, coords); 
	write_imagef(RGBMapBumpSwap, coords, RGBIn);
		
	uint4 labelIn = read_imageui(LabelsMask, smp, coords);
	labelIn.x = labelIn.x/255;
	labelIn.y = labelIn.y/255;
	labelIn.z = labelIn.z/255;
	int flag = labelIn.z + labelIn.y*(2+2*labelIn.z) + labelIn.x*(3+labelIn.y);
	
	float nmle[3] = {0.0f,0.0f,0.0f};
	float pt[3] = {0.0f,0.0f,0.0f};
	pt[0] = VerticesBS[6*tid];
	pt[1] = VerticesBS[6*tid + 1];
	pt[2] = VerticesBS[6*tid + 2];
	nmle[0] = VerticesBS[6*tid + 3];
	nmle[1] = VerticesBS[6*tid + 4];
	nmle[2] = VerticesBS[6*tid + 5];
	
	#pragma unroll
	for (int k = 1; k < NB_BS; k++) {
		// This blended normal is not really a normal since it may be not normalized
		nmle[0] = nmle[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 3] * BlendshapeCoeff[k];
		nmle[1] = nmle[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 4] * BlendshapeCoeff[k];
		nmle[2] = nmle[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 5] * BlendshapeCoeff[k];

		pt[0] = pt[0] + VerticesBS[k*6*n_bump*m_bump + 6*tid] * BlendshapeCoeff[k];
		pt[1] = pt[1] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 1] * BlendshapeCoeff[k];
		pt[2] = pt[2] + VerticesBS[k*6*n_bump*m_bump + 6*tid + 2] * BlendshapeCoeff[k];
	}

	int p_indx[2];
	
	float pt_T[3];
	pt_T[0] = pt[0] * Pose[0] + pt[1] * Pose[4] + pt[2] * Pose[8] + Pose[12];
	pt_T[1] = pt[0] * Pose[1] + pt[1] * Pose[5] + pt[2] * Pose[9] + Pose[13];
	pt_T[2] = pt[0] * Pose[2] + pt[1] * Pose[6] + pt[2] * Pose[10] + Pose[14];
	
	float nmle_T[3];
	nmle_T[0] = nmle[0] * Pose[0] + nmle[1] * Pose[4] + nmle[2] * Pose[8];
	nmle_T[1] = nmle[0] * Pose[1] + nmle[1] * Pose[5] + nmle[2] * Pose[9];
	nmle_T[2] = nmle[0] * Pose[2] + nmle[1] * Pose[6] + nmle[2] * Pose[10];
	
	float fact_BP = 1000.0f;
	float bum_val = bumpIn.x;
	float d = bum_val/fact_BP;
	float maskIn = bumpIn.y; 
	
	float4 VMapInn = {pt[0] + d*nmle[0], pt[1] + d*nmle[1], pt[2] + d*nmle[2], 0.0f};
		
	if (bumpIn.w == -1.0f && bumpIn.y == 0.0f) {
		write_imagef(VMapBump, coords, VMapInn);
		write_imagef(BumpSwap, coords, bumpIn);
		return;
	}
	
	if (nmle_T[2] < 0.0f /*|| maskIn > 99.0f*/) {
		if (maskIn > 0.0f) {
			write_imagef(VMapBump, coords, VMapInn);
		} else {
			write_imagef(VMapBump, coords, zero);
		}
		return;
	}
	
	float4 NMapIn = read_imagef(NMapBump, smp, coords);
	float Tnmle[3];
	Tnmle[0] = NMapIn.x * Pose[0] + NMapIn.y * Pose[4] + NMapIn.z * Pose[8];
	Tnmle[1] = NMapIn.x  * Pose[1] + NMapIn.y * Pose[5] + NMapIn.z * Pose[9];
	Tnmle[2] = NMapIn.x  * Pose[2] + NMapIn.y * Pose[6] + NMapIn.z * Pose[10];
	if (Tnmle[0] == 0.0f && Tnmle[1] == 0.0f && Tnmle[2] == 0.0f) {
		Tnmle[0] = nmle_T[0];
		Tnmle[1] = nmle_T[1];
		Tnmle[2] = nmle_T[2];
	}
	
	float min_dist = 1000000000.0f;
	float best_state = -1.0f;
	float fact_curr = round(maskIn) == 0.0f ? 1.0f : min(5.0f, round(maskIn)+1.0f);
	float pos[3];
	
	//summit 1
	d = (bum_val - (50.0f / fact_curr)) / fact_BP;
	pos[0] = pt_T[0] + d*nmle_T[0];
	pos[1] = pt_T[1] + d*nmle_T[1];
	pos[2] = pt_T[2] + d*nmle_T[2];
	// Project the point onto the depth image
	float s1[2];
	s1[0] = convert_float(min(m - 1, max(0, convert_int(round((pos[0] / fabs(pos[2]))*calib[0] + calib[2])))));
	s1[1] = convert_float(min(n - 1, max(0, convert_int(round((pos[1] / fabs(pos[2]))*calib[1] + calib[3])))));
	
	//summit 2
	d = (bum_val + (50.0f / fact_curr)) / fact_BP;
	pos[0] = pt_T[0] + d*nmle_T[0];
	pos[1] = pt_T[1] + d*nmle_T[1];
	pos[2] = pt_T[2] + d*nmle_T[2];
	
	// Project the point onto the depth image
	float s2[2];
	s2[0] = convert_float(min(m - 1, max(0, convert_int(round((pos[0] / fabs(pos[2]))*calib[0] + calib[2])))));
	s2[1] = convert_float(min(n - 1, max(0, convert_int(round((pos[1] / fabs(pos[2]))*calib[1] + calib[3])))));
	
	float length = sqrt((s1[0]-s2[0])*(s1[0]-s2[0]) + (s1[1]-s2[1])*(s1[1]-s2[1]));
	
	float dir[2];
	dir[0] = (s2[0]-s1[0])/length;
	dir[1] = (s2[1]-s1[1])/length;
	
	d = bum_val / fact_BP;
	pos[0] = pt_T[0] + d*nmle_T[0];
	pos[1] = pt_T[1] + d*nmle_T[1];
	pos[2] = pt_T[2] + d*nmle_T[2];
	
	/*if (j == 150 && i == 225) {
		printf("length: %f \n", length);
		printf("bumpIn: %f %f %f \n", bumpIn.x, bumpIn.y, bumpIn.z);
		printf("nmle_T: %f %f %f \n", nmle_T[0], nmle_T[1], nmle_T[2]);
	}*/
	//printf("length: %f \n", length);
	//printf("%f %f %f \n", bumpIn.x, bumpIn.y, bumpIn.z);
	float4 ptIn;
	float4 nmleIn;
	uint4 flagIn;
	
	float thresh_dist = round(maskIn) == 0.0f ? 0.06f : 0.01f;
	
	//#pragma unroll
	for (float lambda = 0.0f; lambda <= length; lambda += 1.0f) {
		int k = n-1-convert_int(round(s1[1]+ lambda*dir[1]));
		int l = convert_int(round(s1[0] + lambda*dir[0]));
		
		if (k < 0 || k > n - 1 || l < 0 || l > m - 1)
			continue;
			
		int size = 1;
		int ll = max(0, (int)k-size);
		int ul = min(n, (int)k+size+1);
		int lr = max(0, (int)l-size);
		int ur = min(m, (int)l+size+1);
		
		for (int kk = ll; kk < ul; kk++) {
			for (int lk = lr; lk < ur; lk++) {
				ptIn = read_imagef(VMap, smp, (int2){lk, kk}); // (x,y,z, flag)
				nmleIn = read_imagef(NMap, smp, (int2){lk, kk});
				
				if (nmleIn.x == 0.0f && nmleIn.y == 0.0f && nmleIn.z == 0.0f)
					continue;

				//compute distance of point to the normal
				float u_vect[3];
				u_vect[0] = ptIn.x - pt_T[0];
				u_vect[1] = ptIn.y - pt_T[1];
				u_vect[2] = ptIn.z - pt_T[2];

				float proj = u_vect[0] * nmle_T[0] + u_vect[1] * nmle_T[1] + u_vect[2] * nmle_T[2];
				float v_vect[3];
				v_vect[0] = u_vect[0] - proj * nmle_T[0];
				v_vect[1] = u_vect[1] - proj * nmle_T[1];
				v_vect[2] = u_vect[2] - proj * nmle_T[2];
				float dist = sqrt((ptIn.x - pos[0]) * (ptIn.x - pos[0]) + (ptIn.y- pos[1]) * (ptIn.y - pos[1]) + (ptIn.z - pos[2]) * (ptIn.z - pos[2]));
				float dist_to_nmle = sqrt(v_vect[0] * v_vect[0] + v_vect[1] * v_vect[1] + v_vect[2] * v_vect[2]);
				float dist_angle = Tnmle[0] * nmleIn.x + Tnmle[1] * nmleIn.y + Tnmle[2] * nmleIn.z;
				bool valid = (flag == 0) || (flag == convert_int(ptIn.w));

				/*if (j == 150 && i == 225) {
					printf("pixel: %d, %d \n", kk, lk);
					printf("dist_to_nmle: %f \n", dist_to_nmle);
					printf("dist: %f\n", dist);
					printf("dist_angle: %f\n", dist_angle);
				}*/
	
				if (dist_to_nmle < min_dist && dist_angle > 0.7f && /*valid &&*/ dist < thresh_dist) {
					min_dist = dist_to_nmle;
					best_state = proj * fact_BP;
				}
			}
		}
 	}
				
	VMapInn.x = pt[0] + d*nmle[0];
	VMapInn.y = pt[1] + d*nmle[1];
	VMapInn.z = pt[2] + d*nmle[2];
	if (best_state == -1.0f || min_dist > 0.01f) {
		
		//Test for visibility violation
		p_indx[0] = min(m - 1, max(0, convert_int(round((VMapInn.x / fabs(VMapInn.z))*calib[0] + calib[2]))));
		p_indx[1] = n - 1 - min(n - 1, max(0, convert_int(round((VMapInn.y / fabs(VMapInn.z))*calib[1] + calib[3]))));
		ptIn = read_imagef(VMap, smp, (int2){p_indx[0], p_indx[1]});
		
		if (ptIn.z < (VMapInn.z - 0.05f)) { //visibility violation
			if (maskIn < 1.0f) {
				write_imagef(VMapBump, coords, zero);
				bumpIn.x = 0.0f;
				bumpIn.y = 0.0f;
				write_imagef(BumpSwap, coords, bumpIn);
			} else {
				write_imagef(VMapBump, coords, VMapInn);
				bumpIn.y = maskIn - 1.0f;
				write_imagef(BumpSwap, coords, bumpIn);
			}
			return;
		}
	
		if (maskIn > 0.0f) {
			write_imagef(VMapBump, coords, VMapInn);
		} else {
			write_imagef(VMapBump, coords, zero);
		}
		return;
	}

	float weight = 1.0f; //0.003f/(0.0001f + min_dist*min_dist); //maskIn == 0.0f ? 0.1f : Tnmle[2];
	//weight = weight*weight;
	float new_bump = (weight*best_state + bum_val*maskIn) / (maskIn + weight);
	float4 bumpOut = {new_bump, min(100.0f, maskIn + weight), bumpIn.z, 1.0f};
	if (maskIn < 2000.0f) {
		write_imagef(BumpSwap, coords, bumpOut);
	} else {
		new_bump = bum_val;
	}

	//Get color
	float p1[3];
	d = new_bump / fact_BP;
	p1[0] = pt_T[0] + d*nmle_T[0];
	p1[1] = pt_T[1] + d*nmle_T[1];
	p1[2] = pt_T[2] + d*nmle_T[2];

	p_indx[0] = min(m - 1, max(0, convert_int(round((p1[0] / fabs(p1[2]))*calib[0] + calib[2]))));
	p_indx[1] = n - 1 - min(n - 1, max(0, convert_int(round((p1[1] / fabs(p1[2]))*calib[1] + calib[3]))));
	
	uint4 pixelRGB = read_imageui(RGBMap, smp, (int2){p_indx[0],  p_indx[1]});
	ptIn = read_imagef(VMap, smp, (int2){p_indx[0],  p_indx[1]});
	float4 RGBMapIn = read_imagef(RGBMapBump, smp, coords);
	float4 RGBMapOut;
	if ((ptIn.x != 0.0f || ptIn.y != 0.0f || ptIn.z != 0.0f) && maskIn < 2000.0f) {
		RGBMapOut.x = (weight*convert_float(pixelRGB.z) + RGBMapIn.x*maskIn) / (maskIn + weight); 
		RGBMapOut.y = (weight*convert_float(pixelRGB.y) + RGBMapIn.y*maskIn) / (maskIn + weight); 
		RGBMapOut.z = (weight*convert_float(pixelRGB.x) + RGBMapIn.z*maskIn) / (maskIn + weight); 
		write_imagef(RGBMapBumpSwap, coords, RGBMapOut);
	}

	VMapInn.x = pt[0] + d*nmle[0];
	VMapInn.y = pt[1] + d*nmle[1];
	VMapInn.z = pt[2] + d*nmle[2];
	if (maskIn + weight > 0.0) {
		write_imagef(VMapBump, coords, VMapInn);
		//printf("%f %f %f", bumpIn.x, bumpIn.y, bumpIn.z);
	} else {
		write_imagef(VMapBump, coords, zero);
	}
}