#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include <getopt.h>

using namespace std;

struct Node {
	double stock_price,option_price;
};
int main(int argc, char *argv[]) {
    int c;
    double p, up, down;
	int step;
	double stock_price, volatility, risk_free_rate;
	double exercise_price, expiration_time ;

	// parse the command line argument
    while ((c = getopt(argc, argv, "s:v:r:e:x:p:")) >= 0) {
    	switch (c) {
    		case 's': step = atoi(optarg); break;
    		case 'v': sscanf(optarg, "%lf", &volatility); break;
    		case 'r': sscanf(optarg, "%lf", &risk_free_rate); break;
    		case 'e': sscanf(optarg, "%lf", &expiration_time); break;
    		case 'x': sscanf(optarg, "%lf", &exercise_price); break;
    		case 'p': sscanf(optarg, "%lf", &stock_price); break;
    		default: 
    			fprintf(stderr, "unrecognized command line option\n");
    			exit(1);
    	}
    }
	
    int time_arr_size = step + 2;
	int node_arr_size = (step+2)*(step+1)/2 + 1;
	printf("Allocated time array with size %d and node array with size %d\n",
		time_arr_size, node_arr_size);

	Node *node = (Node*) malloc(sizeof(Node) * node_arr_size);
	int *time = (int*) malloc(sizeof(int) * time_arr_size);

	cout << "Stock Price	Volatility	Risk-free Rate	Time to Expiration	Exercise Time	Step:" << endl;
	//cin >> node[1].stock_price >> volatility >> risk_free_rate >> expiration_time >> exercise_price >> step;
	printf("Stock Price:\t\t%.6f\n", stock_price);
	printf("Volatility:\t\t%.6f\n", volatility);
	printf("Risk Free Rate:\t\t%.6f\n", risk_free_rate);
	printf("Expiration Time:\t%.6f\n", expiration_time);
	printf("Exercise Price:\t\t%.6f\n", exercise_price);

	node[1].stock_price = stock_price;

	double time_delta = expiration_time / step;
	double time_sqrt = sqrt(time_delta);
	up = exp(volatility * time_sqrt);
	down = 1 / up;
	double a = exp(risk_free_rate * time_delta);
	p = (a - down) / (up - down);
	step ++;

	cout << p << endl;
	time[0] = 1;
	for (int i = 1; i <= step; i++ )
		time[i] = time[i-1] + i - 1;
	//printf("Time index:\n");
	//for (int i = 1; i <= step; i++) {
	//	printf("time %3d = %5d\n", i, time[i]);
	//}

	for (int i = 2; i <= step; i++) {
		node[time[i]].stock_price = node[time[i-1]].stock_price * up;
		for (int j = 1; j < i; j++) {
			node[time[i]+j].stock_price = 
				node[time[i-1] + j - 1].stock_price * down;
		}
	}

	for (int i = 0; i < step; i++) {
		node[time[step] + i].option_price = 
			-node[time[step] + i].stock_price + exercise_price;
		if (node[time[step] + i].option_price < 0) 
			node[time[step] + i].option_price = 0;
	}


	for (int i = step - 1; i >=1; i--) {
		for (int j = 0; j < i; j++) {
			node[time[i] + j].option_price = 
				(p * node[time[i + 1]+ j].option_price + 
				(1 - p) * node[time[i + 1] + j + 1].option_price) 
				/ a;
			if (node[time[i] + j].option_price < 0) 
				node[time[i] + j].option_price = 0;
		}
	}

	cout << node[1].option_price << endl;
	//for (int i = 0; i <= step; i++) {
	//	for (int j = 0; j < i; j++) {
	//		printf("%3d: %.6f %.6f\n", time[i]+j, node[time[i]+j].stock_price, node[time[i]+j].option_price);
	//	}
	//}
	//system("pause");
	return 0;
}
