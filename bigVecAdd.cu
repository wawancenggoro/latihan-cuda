#include <cuda.h>
#include <time.h>
#include <stdio.h>

__global__ void vecAddKernel(float* A, float* B, float* C, unsigned long n){
	unsigned long i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<n){
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float* A, float* B, float* C, unsigned long n)
{
	unsigned long size = n * sizeof(float);
	static float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);


	clock_t start, end;
	double cpu_time_used;

	start = clock();

	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %f s", cpu_time_used);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	printf("\nFirst 5 values: ");
	for (unsigned long i = 0; i < 5; i++) {
		printf("%.2f ", B[i]);
	}
	printf("\nLast 5 values: ");
	for (unsigned long i = 0; i < 5; i++) {
		printf("%.2f ", B[999994+i]);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
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