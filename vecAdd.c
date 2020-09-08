#include <time.h>
#include <stdio.h>

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	for (int i = 0; i < n; i++) {
		h_C[i] = h_A[i] + h_B[i];
	}

	// print result
	printf("\nA: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", h_A[i]);
	}
	printf("\nB: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", h_B[i]);
	}
	printf("\n-------------------------------------");
	printf("\nC: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", h_C[i]);
	}
	printf("\n-------------------------------------");
}

int main() {
	int N = 5;
	static float h_A[5];
	static float h_B[5];
	static float h_C[5];

	for (int i = 0; i < N; i++) {
		h_A[i] = i%2;
		h_B[i] = i%3;
	}

	vecAdd(h_A, h_B, h_C, N);
}
