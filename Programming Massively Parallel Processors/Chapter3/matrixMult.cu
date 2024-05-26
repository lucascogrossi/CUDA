#include <iostream>
#include <cstdlib>
#include <cassert>

inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

__global__ void matrixMultKernel(float* A, float* B, float* C, int N) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < N) && (col < N)) {
        float sum = 0.0f;

        for (unsigned int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


void initMatrix(float* a, int N) {
    for(unsigned int i = 0; i < N * N; ++i)
        a[i] = rand() % 100;
}

void matrixMult(float* a_h, float* b_h, float* c_h, int N) {
    float *a_d, *b_d, *c_d;
    size_t size = sizeof(float) * N * N;

    checkCuda( cudaMalloc((void**) &a_d, size) );
    checkCuda( cudaMalloc((void**) &b_d, size) );
    checkCuda( cudaMalloc((void**) &c_d, size ));

    checkCuda( cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice) );

    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    matrixMultKernel<<<numBlocks, numThreadsPerBlock>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();

    checkCuda( cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(a_d) );
    checkCuda( cudaFree(b_d) );
    checkCuda( cudaFree(c_d) );
}

int main() {

    int N = 1 << 10;
    size_t size = N * N * sizeof(float);

    float *a, *b, *c;

    a = (float*) malloc(size);
    b = (float*) malloc(size);
    c = (float*) malloc(size);

    initMatrix(a, N);
    initMatrix(b, N);

    matrixMult(a, b, c, N);

    free(a);
    free(b);
    free(c);
}