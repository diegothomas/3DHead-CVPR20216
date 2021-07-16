__kernel void InitKernel(__global int *parents, int n /*height*/, int m /*width*/) {

	int i = get_global_id(0); /*height*/
	int j = get_global_id(1); /*width*/
	int tid = i*m + j;
	
	if (i > n-1 || j > m-1)
		return;
		
	parents[100*tid] = -1;
}