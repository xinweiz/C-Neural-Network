#pragma once
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<time.h>
using namespace std;

class neural_XOR {
public:
	double f(double x) { return 1.0 / (1.0 + exp(-x)); };
	double fprime(double x) { return f(x)*(1 - f(x)); };
	double geterr() { return err; };
	void set_rate(double x) { rate = x; }
	void set_err(double x) { target_err = x; }
	void set_initialweight();
	void BP();
private:
	double x[4][2] = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	double t[4] = { 0, 1, 1, 0 };
	double w[2], v[2][2], b[2], h[4][2], y[4];
	double delta_w[2] = {0};
	double delta_v[2][2] = {0};
	double delta_b[2] = {0};
	double err = 0;
	double rate, target_err;
};