__kernel void VmapKernel(__read_only image2d_t depth, __read_only image2d_t segments, __write_only image2d_t vmap, __constant float *calib, int Kinect, int n /*height*/, int m /*width*/) {

	unsigned int i = get_global_id(0); /*height*/
	unsigned int j = get_global_id(1); /*width*/
	unsigned int tid = i*m + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	if (i > n-1 || j > m-1)
		return;
		
	uint4 pixel = read_imageui(depth, smp, coords);
	
	float4 pt;
	float d;
	if (Kinect == 0) {
		pt.x = convert_float(pixel.x) / 6000.0f - 5.f;
		pt.y = convert_float(pixel.y) / 6000.0f - 5.f;
		pt.z = -convert_float(pixel.z) / 10000.0f, 0.0f;
	} else {
		d = convert_float(pixel.z);
		pt.x = d == 0.0 ? 0.0 : (calib[9]/calib[10]) * d * ((convert_float(j)-calib[2])/calib[0]);
		pt.y = d == 0.0 ? 0.0 : (calib[9]/calib[10]) * d * ((convert_float(n-1-i)-calib[3])/calib[1]);
		pt.z = d == 0.0 ? 0.0 : -(calib[9]/calib[10]) * d;
	}
	
	pixel = read_imageui(segments, smp, coords);
	if (pixel.z == 0 && pixel.y == 255 && pixel.x == 255)
		pt.w = 1.0f; 
	if (pixel.z == 0 && pixel.y  == 0 && pixel.x == 255)
		pt.w = 2.0f; 
	if (pixel.z == 255 && pixel.y  == 255 && pixel.x == 0)
		pt.w = 3.0f; 
	if (pixel.z == 0 && pixel.y  == 255 && pixel.x == 0)
		pt.w = 4.0f; 
	if (pixel.z == 255 && pixel.y  == 0 && pixel.x == 255)
		pt.w = 5.0f; 
	if (pixel.z == 255 && pixel.y  == 0 && pixel.x == 0)
		pt.w = 6.0f; 
	
	write_imagef(vmap, coords, pt);
}