#include <time.h>
#include <stdio.h>

void vecAdd(float* h_A, float* h_B, float* h_C, unsigned long n)
{
	clock_t start, end;
	double cpu_time_used;

	start = clock();
	for (unsigned long i = 0; i < n; i++) {
		h_C[i] = h_A[i] + h_B[i];
	}
	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("\nElapsed time: %f s", cpu_time_used);

	printf("\nFirst 5 values: ");
	for (unsigned long i = 0; i < 5; i++) {
		printf("%.2f ", h_C[i]);
	}
	printf("\nLast 5 values: ");
	for (unsigned long i = 0; i < 5; i++) {
		printf("%.2f ", h_C[999994+i]);
	}
}

int main() {
	unsigned long N = 1000000;
	static float h_A[1000000];
	static float h_B[1000000];
	static float h_C[1000000];

	for (unsigned long i = 0; i < N; i++) {
		h_A[i] = i%2;
		h_B[i] = i%3;
	}

	vecAdd(h_A, h_B, h_C, N);
}
