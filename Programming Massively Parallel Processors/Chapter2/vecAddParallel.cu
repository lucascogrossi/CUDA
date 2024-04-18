#include <stdio.h>
#include <stdlib.h>

#define N 1000

__global__
void vecAddkernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddkernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    float *v1, *v2, *v3;

    v1 = (float*) malloc(sizeof(float) * N);
    v2 = (float*) malloc(sizeof(float) * N);
    v3 = (float*) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        v1[i] = (float) i;
        v2[i] = (float) i;
        v3[i] = 0;
    }

    vecAdd(v1, v2, v3, N);
    
    for (int i = 0; i < N; i++) {
        if (v3[i] != v1[i] + v2[i]) {
            printf("Error\n");
            return 1;
        }   
    }
    printf("Successful vector addition\n");
    return 0;

}