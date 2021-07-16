__kernel void NmapKernel(__read_only image2d_t vmap, __write_only image2d_t nmap, int n, int m) {

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
    float4 n_p = {0.0f,0.0f,0.0f,1.0f};
    float n_p1 [3];
    float n_p2 [3];
    float n_p3 [3];
    float n_p4 [3];
    float norm_n;
    unsigned short n_tot = 0;
        
	p1 = read_imagef(vmap, smp, coords);
    if ( i < 1 || j < 1 || p1.z == 0.0f) {
		write_imagef(nmap, coords, n_p);
        return;
    }

    n_p1[0] = 0.0f; n_p1[1] = 0.0f; n_p1[2] = 0.0f;
    n_p2[0] = 0.0f; n_p2[1] = 0.0f; n_p2[2] = 0.0f;
    n_p3[0] = 0.0f; n_p3[1] = 0.0f; n_p3[2] = 0.0f;
    n_p4[0] = 0.0f; n_p4[1] = 0.0f; n_p4[2] = 0.0f;

    ////////////////////////// Triangle 1 /////////////////////////////////

	p2 = read_imagef(vmap, smp, (int2){j, i+1});
	
	p3 = read_imagef(vmap, smp, (int2){j+1, i});

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

	p2 = read_imagef(vmap, smp, (int2){j+1, i});
	
	p3 = read_imagef(vmap, smp, (int2){j, i-1});

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

	p2 = read_imagef(vmap, smp, (int2){j, i-1});
	
	p3 = read_imagef(vmap, smp, (int2){j-1, i});
	
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

	p2 = read_imagef(vmap, smp, (int2){j-1, i});
	
	p3 = read_imagef(vmap, smp, (int2){j, i+1});

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
		write_imagef(nmap, coords, n_p);
        return;
    }

    n_p.x = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0])/((float)n_tot);
    n_p.y = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1])/((float)n_tot);
    n_p.z = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2])/((float)n_tot);

    norm_n = sqrt(n_p.x*n_p.x + n_p.y*n_p.y + n_p.z*n_p.z);

    if (norm_n != 0.0f) {
		n_p.x = n_p.x/norm_n;
		n_p.y = n_p.y/norm_n;
		n_p.z = n_p.z/norm_n;
		write_imagef(nmap, coords, n_p);
    } else {
		n_p.x = 0.0f;
        n_p.y = 0.0f;
        n_p.z = 0.0f;
		write_imagef(nmap, coords, n_p);
	}
}

