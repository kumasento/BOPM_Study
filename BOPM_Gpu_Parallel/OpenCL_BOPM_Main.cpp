
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <getopt.h>

#include "opencl.h"

#define ALLOC_R_BUFFER(buf, size) 	AllocBuffer(buf, (size), MEM_READ_FLAGS)
#define ALLOC_RW_BUFFER(buf, size) 	AllocBuffer(buf, (size), MEM_READ_WRITE_FLAGS)

using namespace std;

struct BopmNode {
	double sp; // stock price
	double op; // option price
};

int main(int argc, char *argv[]) {
	int c;
	int platform_id = 1;
	int device_id = 0;
	int device_type = CL_DEVICE_TYPE_GPU;
	int iter_time = 0;
	int level = 1;

	OclModule ocl;
	ocl.InitModule(platform_id, device_id, device_type);

	// read configure from standard input
	double p, up, down;
	int step;
	double stock_price, volatility, risk_free_rate;
	double exercise_price, expiration_time;

	printf("Running algorithm ...\n");
	// first get the step number
	while ((c = getopt(argc, argv, "s:v:r:e:x:p:i:l:")) >= 0) {
    	switch (c) {
    		case 's': step = atoi(optarg); break;
    		case 'v': sscanf(optarg, "%lf", &volatility); break;
    		case 'r': sscanf(optarg, "%lf", &risk_free_rate); break;
    		case 'e': sscanf(optarg, "%lf", &expiration_time); break;
    		case 'x': sscanf(optarg, "%lf", &exercise_price); break;
    		case 'p': sscanf(optarg, "%lf", &stock_price); break;
    		case 'i': iter_time = atoi(optarg); break;
    		case 'l': level = atoi(optarg); break;
    		default: 
    			fprintf(stderr, "unrecognized command line option\n");
    			exit(1);
    	}
    }

	// and then use #step to build the memory for time and node
	int time_arr_size = step + 2;
	int node_arr_size = (step+2)*(step+1)/2 + 1;
	printf("Allocated time array with size %d and node array with size %d\n",
		time_arr_size, node_arr_size);
	
	printf("Stock Price:\t\t%.6f\n", stock_price);
	printf("Volatility:\t\t%.6f\n", volatility);
	printf("Risk Free Rate:\t\t%.6f\n", risk_free_rate);
	printf("Expiration Time:\t%.6f\n", expiration_time);
	printf("Exercise Price:\t\t%.6f\n", exercise_price);

	double time_delta = expiration_time / step;
	double time_sqrt = sqrt(time_delta);
	up = exp(volatility * time_sqrt);
	down = 1 / up;
	double a = exp(risk_free_rate * time_delta);
	p = (a - down) / (up - down);
	cout << p << endl;

	const char *kernel_name = "OpenCL_BOPM_Kernel.cl";
	ocl.InitProgram(kernel_name);
	ocl.InitKernel("BopmComputeKernel");

	BopmNode *node = (BopmNode*) malloc(sizeof(BopmNode)*node_arr_size);

	cl_mem cl_node = ocl.ALLOC_RW_BUFFER(node, sizeof(BopmNode)*node_arr_size);
	// here we just have one workgroup
	size_t global_size[] = { step + 1};
	size_t local_size[] = { step + 1};

	printf("workgroup size: %d\n", step + 1);

	ocl.kernels[0].SetKernel(global_size,local_size,1);
	ocl.SetKernelArgs(0, &cl_node);
	ocl.SetKernelArgs(0, &up);
	ocl.SetKernelArgs(0, &down);
	ocl.SetKernelArgs(0, &exercise_price);
	ocl.SetKernelArgs(0, &p);
	ocl.SetKernelArgs(0, &stock_price);
	ocl.SetKernelArgs(0, &a);
	ocl.SetKernelArgs(0, &level);
	ocl.RunKernel(0);
	ocl.ReadBuffer(node, sizeof(BopmNode)*node_arr_size, cl_node);

	double duration = 0.0;
	for (int i = 0; i < iter_time; i++) {
		cl_event event = ocl.RunKernel(0);
		duration += ExecutionTime(event);
	}
	printf("Time %.6f ms\n", (double)duration/iter_time);

	//for (int i = 0; i < node_arr_size; i++) {
	//	printf("%d: %6.6f %6.6f\n", i, node[i].sp, node[i].op);
	//}

	printf("Answer: %f\n", node[1].op);

    return 0;
}
