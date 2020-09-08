#include <cuda.h>
#include <stdio.h>

#define TILE_WIDTH 2

__global__ void matMulKernel(float* d_N, float* d_M, float* d_P, int Width){
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;

	// Loop over the d_M and d_N tiles required to compute d_P element
	for (int m = 0; m < Width/TILE_WIDTH; ++m) {// Coolaborative loading of d_M and d_N tiles into shared memory
		Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();
		
		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	d_P[Row*Width + Col] = Pvalue; 
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