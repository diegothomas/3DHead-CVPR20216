__kernel void NmapBumpKernel(__read_only image2d_t VMapBump, __write_only image2d_t NMapBump, int n, int m) {

	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	unsigned int tid = i*m + j;
	int2 coords = (int2){get_global_id(1), get_global_id(0)};
	const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	
	if ( i > n-2 || j > m-2)  {
        return;
    }	
	
	float4 p1;
	float4 p2;
	float4 p3;
	float n_p[3];
	float n_p1[3];
	float n_p2[3];
	float n_p3[3];
	float n_p4[3];
	float norm_n;
	p1 = read_imagef(VMapBump, smp, coords);
	float4 NOut = {0.0f,0.0f,0.0f,0.0f};
        
    if ( i < 1 || j < 1 || (p1.x == 0.0f && p1.y == 0.0f && p1.z == 0.0f)) {
		write_imagef(NMapBump, coords, NOut);
        return;
    }

	unsigned short n_tot = 0;

	n_p1[0] = 0.0; n_p1[1] = 0.0; n_p1[2] = 0.0;
	n_p2[0] = 0.0; n_p2[1] = 0.0; n_p2[2] = 0.0;
	n_p3[0] = 0.0; n_p3[1] = 0.0; n_p3[2] = 0.0;
	n_p4[0] = 0.0; n_p4[1] = 0.0; n_p4[2] = 0.0;

	////////////////////////// Triangle 1 /////////////////////////////////
	p2 = read_imagef(VMapBump, smp, (int2){j,i+1});
	p3 = read_imagef(VMapBump, smp, (int2){j+1,i});

	if (p2.z != 0.0f && p3.z != 0.0f) {
        n_p1[0] = (p2.y-p1.y)*(p3.z-p1.z) - (p2.z-p1.z)*(p3.y-p1.y);
        n_p1[1] = (p2.z-p1.z)*(p3.x-p1.x) - (p2.x-p1.x)*(p3.z-p1.z);
        n_p1[2] = (p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x);

        norm_n = (n_p1[0]*n_p1[0] + n_p1[1]*n_p1[1] + n_p1[2]*n_p1[2]);

        if (norm_n != 0.0f) {
            n_p1[0] = n_p1[0] / sqrt(norm_n);
            n_p1[1] = n_p1[1] / sqrt(norm_n);
            n_p1[2] = n_p1[2] / sqrt(norm_n);

            n_tot++;
        }
    }

	////////////////////////// Triangle 2 /////////////////////////////////

	p2 = read_imagef(VMapBump, smp, (int2){j+1, i});
	p3 = read_imagef(VMapBump, smp, (int2){j, i-1});

    if (p2.z != 0.0f && p3.z != 0.0f) {
		n_p2[0] = (p2.y-p1.y)*(p3.z-p1.z) - (p2.z-p1.z)*(p3.y-p1.y);
        n_p2[1] = (p2.z-p1.z)*(p3.x-p1.x) - (p2.x-p1.x)*(p3.z-p1.z);
        n_p2[2] = (p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x);

        norm_n = (n_p2[0]*n_p2[0] + n_p2[1]*n_p2[1] + n_p2[2]*n_p2[2]);

        if (norm_n != 0.0f) {
            n_p2[0] = n_p2[0] / sqrt(norm_n);
            n_p2[1] = n_p2[1] / sqrt(norm_n);
            n_p2[2] = n_p2[2] / sqrt(norm_n);

            n_tot++;
        }
    }

    ////////////////////////// Triangle 3 /////////////////////////////////

	p2 = read_imagef(VMapBump, smp, (int2){j, i-1});
	p3 = read_imagef(VMapBump, smp, (int2){j-1, i});
	
    if (p2.z != 0.0f && p3.z != 0.0f) {
		n_p3[0] = (p2.y-p1.y)*(p3.z-p1.z) - (p2.z-p1.z)*(p3.y-p1.y);
        n_p3[1] = (p2.z-p1.z)*(p3.x-p1.x) - (p2.x-p1.x)*(p3.z-p1.z);
        n_p3[2] = (p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x);

        norm_n = (n_p3[0]*n_p3[0] + n_p3[1]*n_p3[1] + n_p3[2]*n_p3[2]);

        if (norm_n != 0.0f) {
            n_p3[0] = n_p3[0] / sqrt(norm_n);
            n_p3[1] = n_p3[1] / sqrt(norm_n);
            n_p3[2] = n_p3[2] / sqrt(norm_n);

            n_tot++;
        }
    }

    ////////////////////////// Triangle 4 /////////////////////////////////

	p2 = read_imagef(VMapBump, smp, (int2){j-1, i});
	p3 = read_imagef(VMapBump, smp, (int2){j, i+1});

    if (p2.z != 0.0f && p3.z != 0.0f) {
		n_p4[0] = (p2.y-p1.y)*(p3.z-p1.z) - (p2.z-p1.z)*(p3.y-p1.y);
        n_p4[1] = (p2.z-p1.z)*(p3.x-p1.x) - (p2.x-p1.x)*(p3.z-p1.z);
        n_p4[2] = (p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x);

        norm_n = (n_p4[0]*n_p4[0] + n_p4[1]*n_p4[1] + n_p4[2]*n_p4[2]);

        if (norm_n != 0.0f) {
            n_p4[0] = n_p4[0] / sqrt(norm_n);
            n_p4[1] = n_p4[1] / sqrt(norm_n);
            n_p4[2] = n_p4[2] / sqrt(norm_n);

            n_tot++;
        }
    }

	if (n_tot == 0) {
		write_imagef(NMapBump, coords, NOut);
		return;
	}

	n_p[0] = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0]) / ((float)n_tot);
	n_p[1] = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1]) / ((float)n_tot);
	n_p[2] = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2]) / ((float)n_tot);

	norm_n = sqrt(n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);

	if (norm_n != 0) {
		NOut.x = n_p[0] / norm_n;
		NOut.y = n_p[1] / norm_n;
		NOut.z = n_p[2] / norm_n;
		write_imagef(NMapBump, coords, NOut);
	}
	else {
		write_imagef(NMapBump, coords, NOut);
	}
}

