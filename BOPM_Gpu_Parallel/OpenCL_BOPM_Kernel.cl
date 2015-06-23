
#pragma OPENCL EXTENSION cl_khr_fp64: enable

struct BopmNode {
	double sp; // stock price
	double op; // option price
};

__kernel void BopmComputeKernel(
	__global struct BopmNode 	*node,
	double 						up,
	double 						down,
	double 						exercise_price,
	double 						p,
	double 						stock_price,
	double 						a,
	int 						L
	) {
	int idx = get_global_id(0); // here global id equals to local id
	int N = get_global_size(0); // number of steps
	int start_idx;
	int id;
	if (idx == 0) 
		node[1].sp = stock_price;
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	start_idx = 1;
	for (int i = 1; i < N; i++) {
		id = idx + start_idx + 1;
		if (idx == 0) {
			node[id].sp = node[id-i].sp * up;
		}
		else if (idx > 0 && idx <= i) {
			node[id].sp = node[id-i-1].sp * down;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		start_idx += i + 1;
	}

	id = idx + start_idx - N + 1;
	//printf("%d %d\n", idx, id);
	node[id].op = -node[id].sp + exercise_price;
	if (node[id].op < 0) 
		node[id].op = 0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	start_idx -= 2*N - 1;
	for (int i = N-2; i >= 0; i--) {
		id = idx + start_idx + 1;
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (idx >= 0 && idx <= i) {
			node[id].op = (p * node[id+i+1].op + (1-p) * node[id+i+2].op)/a;
			if (node[id].op < 0)
				node[id].op = 0;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		start_idx -= i;
	}
}