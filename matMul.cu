#include <cuda.h>
#include <stdio.h>

__global__ void matMulKernel(float* M, float* N, float* P, int width){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;

	if ((row < width) && (col < width)) {
		float Pvalue = 0;
		for (int k = 0; k < width; ++k){
			Pvalue += M[k*width+col]*N[row*width+k];
		}
		P[row*width+col] = Pvalue;
	}
}

void matMul(float* A, float* B, float* C, int width)
{
	int size = width * width * sizeof(float);
	static float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 dimGrid(2, 2, 1);
	dim3 dimBlock(2, 2, 1);

	matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	printf("\nA: \n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++){
			printf("%2.0f ", A[i + j*width]);
		}
		printf("\n");
	}
	printf("\nB: \n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++){
			printf("%2.0f ", B[i + j*width]);
		}
		printf("\n");
	}
	printf("\n-------------------------------------");
	printf("\nC: \n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++){
			printf("%2.0f ", C[i + j*width]);
		}
		printf("\n");
	}
	printf("\n-------------------------------------\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	int width = 4;
	static float h_A[16];
	static float h_B[16];
	static float h_C[16];

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			h_A[i + j*width] = (i+j)%2;
			h_B[i + j*width] = (i+j)%3;
		}
	}
	matMul(h_A, h_B, h_C, width);
}