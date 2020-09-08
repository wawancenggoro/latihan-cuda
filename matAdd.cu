#include <cuda.h>
#include <stdio.h>

__global__ void matAddKernel(float* A, float* B, float* C, int width, int height){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;

	int i = col + row*width;

	if(i < width*height){
		C[i] = A[i] + B[i];
	}
}

void matAdd(float* A, float* B, float* C, int width, int height)
{
	int size = width * height * sizeof(float);
	static float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 dimGrid(5, 4, 1);
	dim3 dimBlock(16, 16, 1);

	matAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width, height);

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
	int width = 76;
	int height = 62;
	static float h_A[76*62];
	static float h_B[76*62];
	static float h_C[76*62];

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			h_A[i + j*width] = (i+j)%2;
			h_B[i + j*width] = (i+j)%3;
		}
	}
	matAdd(h_A, h_B, h_C, width, height);
}