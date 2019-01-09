#include"Neural.h"

void recurrence() {
	neural_XOR T;
	int choice;
	cout << "1. Use XOR" << endl;
	cout << "2. Exit" << endl;
	cout << "Please select: ";
	cin >> choice;
	switch (choice) {
	case 1:
		double x, y;
		cout << "Please input the learning rate: ";
		cin >> x;
		T.set_rate(x);
		cout << "Please input the target error: ";
		cin >> y;
		T.set_err(y);
		T.set_initialweight();
		T.BP();
		cout << "------------------------------------" << endl;
		recurrence();
		break;
	case 2:
		return;
		break;
	default:
		cout << "Error, please input again!" << endl;
		recurrence();
		break;
	}
}

int main() {
	recurrence();
	system("pause");
	return 0;
}