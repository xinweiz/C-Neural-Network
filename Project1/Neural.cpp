#include"Neural.h"

void neural_XOR::set_initialweight() {
	srand((unsigned)time(NULL));
	for (int i = 0; i < 2; i++) {
		w[i] = (float)rand() / RAND_MAX * 2 - 1;
		b[i] = (float)rand() / RAND_MAX * 2 - 1;
		for (int j = 0; j < 2; j++)
			v[i][j] = (float)rand() / RAND_MAX * 2 - 1;
	}
	std::cout << "Initial weight        :" << endl;
	std::cout << "W                     : {" << w[0] << ", " << w[1] << "}" << endl;
	std::cout << "V                     : {" << v[0][0] << ", " << v[0][1] << ", " << v[1][0] << ", " << v[1][1] << "}" << endl;
}

void neural_XOR::BP() {
	int n;
	int i = 0;
	while(true) {
		n = i;
		delta_w[2] = { 0 };
		delta_v[2][2] = { 0 };
		delta_b[2] = { 0 };
		err = 0;
		for (int j = 0; j < 4; j++) {
			h[j][0] = f(v[0][0] * x[j][0] + v[0][1] * x[j][1] + b[0]);
			h[j][1] = f(v[1][0] * x[j][0] + v[1][1] * x[j][1] + b[1]);
			y[j] = f(w[0] * h[j][0] + w[1] * h[j][1]);
			err += 0.5 * (t[j] - y[j]) * (t[j] - y[j]);
		}

		if (i == 0)
			std::cout << "First Batch Error     : " << err << endl;
		if (err < target_err)
			break;
		double temp;
		for (int j = 0; j < 4; j++) {
			temp = (t[j] - y[j]) * fprime(w[0] * h[j][0] + w[1] * h[j][1]);
			delta_w[0] += temp * h[j][0];
			delta_w[1] += temp * h[j][1];
			delta_v[0][0] += temp * w[0] * fprime(v[0][0] * x[j][0] + v[0][1] * x[j][1] + b[0]) * x[j][0];
			delta_v[0][1] += temp * w[0] * fprime(v[0][0] * x[j][0] + v[0][1] * x[j][1] + b[0]) * x[j][1];
			delta_v[1][0] += temp * w[1] * fprime(v[1][0] * x[j][0] + v[1][1] * x[j][1] + b[1]) * x[j][0];
			delta_v[1][1] += temp * w[1] * fprime(v[1][0] * x[j][0] + v[1][1] * x[j][1] + b[1]) * x[j][1];
			delta_b[0] += (-1) * temp * w[0] * fprime(v[0][0] * x[j][0] + v[0][1] * x[j][1] + b[0]);
			delta_b[1] += (-1) * temp * w[1] * fprime(v[1][0] * x[j][0] + v[1][1] * x[j][1] + b[1]);
		}
		for (int j = 0; j < 2; j++) {
			w[j] += rate * delta_w[j];
			v[j][0] += rate * delta_v[j][0];
			v[j][1] += rate * delta_v[j][1];
			b[j] += rate * delta_b[j];
		}
		i++;
	}
	std::cout << "Final weight          :" << endl;
	std::cout << "W                     : {" << w[0] << ", " << w[1] << "}" << endl;
	std::cout << "V                     : {" << v[0][0] << ", " << v[0][1] << ", " << v[1][0] << ", " << v[1][1] << "}" << endl;
	std::cout << "Final Error           : " << err << endl;
	std::cout << "The number of batches : " << n << endl;
}