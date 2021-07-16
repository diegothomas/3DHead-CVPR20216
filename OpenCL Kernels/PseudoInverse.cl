#pragma OPENCL EXTENSION cl_amd_printf : enable
#define	NB_BS 28

__kernel void PseudoInverseKernel(__global float *buf, int fact,
								__constant float *Q_inv, __global float *B,
								int nb_lines) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i * get_global_size(1) + j;
	float val;
	
	if (tid > nb_lines-1)
		return;
	
	for (int k = 0; k < NB_BS-1; k++) {
		val = 0.0;
		#pragma unroll
		for (int l = 0; l < NB_BS-1; l++) {
			val = val + Q_inv[k*(NB_BS-1) + l] * B[NB_BS*tid + l];
		}
		buf[(NB_BS-1)*tid + k] = val;
	}
}