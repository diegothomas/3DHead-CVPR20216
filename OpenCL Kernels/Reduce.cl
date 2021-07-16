#pragma OPENCL EXTENSION cl_amd_printf : enable

#define STRIDE 256

void reduce(__local double *buffer, int CTA_SIZE)
{
	int tid = get_local_id(0);
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


__kernel void ReduceKernel (__global double *buf, __global double *output, int length) {

	int beg = get_group_id(0)*length;
	int end = beg + length;

    int tid = get_local_id(0);

	//printf("%d \n", length);
    double sum = 0.0;
    for (int t = beg + tid; t < end; t += STRIDE) {
        sum += buf[t];
	}
	
    __local double smem[STRIDE];

    smem[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

	reduce(smem, STRIDE);
			
	if (tid == 0) {			
		output[get_group_id(0)] = smem[0];
		//printf("%f\n", smem[0]);
	}
}