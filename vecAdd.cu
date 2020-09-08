#include <cuda.h>
#include <time.h>
#include <stdio.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<n){
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float* A, float* B, float* C, int n)
{
	int size = n * sizeof(float);
	static float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size); // allocate d_A
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); // copy A to GPU

	cudaMalloc((void **) &d_B, size); // allocate d_B
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); // copy B to GPU

	cudaMalloc((void **) &d_C, size); // allocate d_C

	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n); // launch kernel

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); // copy d_C to host

	// print result
	printf("\nA: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\nB: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", B[i]);
	}
	printf("\n-------------------------------------");
	printf("\nC: ");
	for (int i = 0; i < 5; i++) {
		printf("%.2f ", C[i]);
	}
	printf("\n-------------------------------------\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
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