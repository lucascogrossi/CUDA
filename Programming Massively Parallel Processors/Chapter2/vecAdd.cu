#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1000

inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

__global__ void vecAddkernel(float* A, float* B, float* C, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);

    checkCuda( cudaMalloc((void**) &A_d, size) );
    checkCuda( cudaMalloc((void**) &B_d, size) );
    checkCuda( cudaMalloc((void**) &C_d, size) );

    checkCuda( cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice) );

    vecAddkernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaDeviceSynchronize();

    checkCuda( cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(A_d) );
    checkCuda( cudaFree(B_d) );
    checkCuda( cudaFree(C_d) );
}

int main() {
    float *v1, *v2, *v3;

    v1 = (float*) malloc(sizeof(float) * N);
    v2 = (float*) malloc(sizeof(float) * N);
    v3 = (float*) malloc(sizeof(float) * N);

    for (unsigned int i = 0; i < N; ++i) {
        v1[i] = rand();
        v2[i] = rand();
        v3[i] = 0;
    }

    vecAdd(v1, v2, v3, N);
  
    for (unsigned int i = 0; i < N; ++i) {
        if (v3[i] != v1[i] + v2[i]) {
            printf("Error\n");
            return 1;
        }   
    }
    printf("Successful vector addition\n");
    return 0;

}