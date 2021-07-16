#pragma OPENCL EXTENSION cl_amd_printf : enable

#define STRIDE 256

void reduce(__local float *buffer, int CTA_SIZE)
{
	int tid = get_local_id(0);
	float val =  buffer[tid];

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


__kernel void Reduce2Kernel (__global float *buf, __global float *output, int length, int length_out) {
 // Assume buf is of size group_size(0)*length
	int a = get_group_id(0);
	int b = 2*get_group_id(1);
    int tid = get_local_id(0);
	
    float sum = 0.0;
	if (2*tid + b*STRIDE < length)
		sum = buf[a*length + 2*tid + b*STRIDE];
	if (2*tid + 1 + b*STRIDE < length)
		sum += buf[a*length + 2*tid + 1 + b*STRIDE];
	
    __local float smem[STRIDE];

    smem[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

	reduce(smem, STRIDE);
			
	if (tid == 0) {			
		output[get_group_id(0)*length_out + get_group_id(1)] = smem[0];
		//if (smem[0] != 0.0)
		//	printf("%f\n", smem[0]);
	}
}