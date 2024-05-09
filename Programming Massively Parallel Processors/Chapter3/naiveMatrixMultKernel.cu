/*
    Parallelization approach: assign one thread to each element
    in the output matrix (C)
*/

__global__ void matrixMultKernel(float* A, float* B, float* C, unsigned int N) {
    
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (unsigned int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;

}

int main() {
    // ...

    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    matrixMultKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);
}