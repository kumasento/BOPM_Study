#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace std;

struct Node {
	double stock_price,option_price;
};
int main() {
	double p, up, down;
	int step;
	double stock_price, volatility, risk_free_rate, exercise_price, expiration_time ;
	Node node[1000000]; 
	int time[10000];

	cout << "Stock Price	Volatility	Risk-free Rate	Time to Expiration	Exercise Time	Step:" << endl;
	cin >> node[1].stock_price >> volatility >> risk_free_rate >> expiration_time >> exercise_price >> step;
	
	double time_delta = expiration_time / step;
	double time_sqrt = sqrt(time_delta);
	up = exp(volatility * time_sqrt);
	down = 1 / up;
	double a = exp(risk_free_rate * time_delta);
	p = (a - down) / (up - down);
	step ++;

	time[0] = 1;
	for (int i = 1; i <= step; i++ )
		time[i] = time[i-1] + i - 1;
	for (int i = 2; i <= step; i++) {
		node[time[i]].stock_price = node[time[i-1]].stock_price * up;
		for (int j = 1; j < i; j++)
			node[time[i]+j].stock_price = node[time[i-1] + j - 1].stock_price * down;
	}

	for (int i = 0; i < step; i++) {
		node[time[step] + i].option_price = -node[time[step] + i].stock_price + exercise_price;
		if (node[time[step] + i].option_price < 0) node[time[step] + i].option_price = 0;
	}


	for (int i = step - 1; i >=1; i--) {
		for (int j = 0; j < i; j++) {
			node[time[i] + j].option_price = (p * node[time[i + 1]+ j].option_price + (1 - p) * node[time[i + 1] + j + 1].option_price) / a;
			if (node[time[i] + j].option_price < 0) node[time[i] + j].option_price = 0;
		}
	}

	cout << node[1].option_price << endl;
	//system("pause");
	return 0;
}