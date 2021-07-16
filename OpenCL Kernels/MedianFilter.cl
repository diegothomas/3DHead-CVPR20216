__kernel void MedianFilterKernel(__write_only image2d_t Bump, __read_only image2d_t BumpSwap, 
								 __write_only image2d_t RGBMapBump, __read_only image2d_t RGBMapBumpSwap, int n, int m) {

	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	if (i > n-1 || j > m-1)
		return;
		
	float4 pix_out = {0.0f,0.0f,0.0f,0.0f};
	float4 color_out = read_imagef(RGBMapBumpSwap, smp, coords);
	float4 pixel_out = read_imagef(BumpSwap, smp, coords);
	pix_out.x = pixel_out.x;
	pix_out.y = pixel_out.y;
	pix_out.z = pixel_out.z;
	pix_out.w = pixel_out.w;
	write_imagef(Bump, coords, pix_out);
	write_imagef(RGBMapBump, coords, color_out);
	
	//if (pix_out.w == -1.0f && pix_out.y != 0.0f)
	//	return;
		
	int size = 1; //pix_out.w == -1.0f ? 5 : 1;
	int ll = max(0, (int)i-size);
	int ul = min(n, (int)i+size+1);
	int lr = max(0, (int)j-size);
	int ur = min(m, (int)j+size+1);
	
	float tab[9];
	
	int count = 0;
	int q = 0;
	float avg = 0.0f;
	float avgR = 0.0f;
	float avgG = 0.0f;
	float avgB = 0.0f;
	float weight = 0.0f;
	float4 pixel_curr, color_curr;
	for (int k = ll; k < ul; k++) {
		for (int l = lr; l < ur; l++) {
			pixel_curr = read_imagef(BumpSwap, smp, (int2){l,k});
			color_curr = read_imagef(RGBMapBumpSwap, smp, (int2){l,k});
			if (pixel_curr.y > 0.0f) {
				q = 0;
				while (q < count && tab[q] > pixel_curr.x)
					q++;
				for (int r = count; r > q; r--)
					tab[r] = tab[r-1];
				tab[q] = pixel_curr.x;
				avg += pixel_curr.x;
				avgR += color_curr.x;
				avgG += color_curr.y;
				avgB += color_curr.z;
				weight += 1.0f;
				count++;
			}
			
		}
	}
	
	if (pix_out.w == -1.0f && pix_out.y == 0.0f) {
		if (weight > 0.0f) {
			pix_out.x = avg/weight;
			pix_out.y = 1.0f;
			color_out.x = avgR/weight;
			color_out.y = avgG/weight;
			color_out.z = avgB/weight;
			write_imagef(RGBMapBump, coords, color_out);
		}
	} else {
		if (count > 0)
			pix_out.x = tab[count/2];
	}
	
	write_imagef(Bump, coords, pix_out);
}