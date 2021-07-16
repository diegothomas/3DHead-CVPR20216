__kernel void BilateralKernel(__read_only image2d_t depth, __write_only image2d_t buff, float sigma_d, float sigma_r, int size, int n, int m) {

	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	if (i > n-1 || j > m-1)
		return;
		
	uint4 pixel_ref = read_imageui(depth, smp, coords);
		
	int ll = max(0, (int)i-size);
	int ul = min(n, (int)i+size+1);
	int lr = max(0, (int)j-size);
	int ur = min(m, (int)j+size+1);
	
	float avg = 0.0;
	float weight = 0.0;
	float w;
	float fact = 0.0;
	for (int k = ll; k < ul; k++) {
		for (int l = lr; l < ur; l++) {
			uint4 pixel_curr = read_imageui(depth, smp, (int2){l,k});
			fact = - (convert_float((i-k)*(i-k) + (j-l)*(j-l)))/(2.0f*sigma_d*sigma_d) - convert_float(pixel_ref.z - pixel_curr.z) * convert_float(pixel_ref.z - pixel_curr.z)/(2.0*sigma_r*sigma_r);
			w = exp(fact);
			
			weight += w;
			avg += w*convert_float(pixel_curr.z);
		}
	}
	
	uint4 pixel_out = {pixel_ref.x,pixel_ref.y,pixel_ref.z,0};
	if (avg < 3000.0) {
		pixel_out.x = 0; pixel_out.y = 0; pixel_out.z = 0;
	} else {
		pixel_out.z = convert_uint(avg/weight);
	}
	
	write_imageui(buff, coords, pixel_out);
}