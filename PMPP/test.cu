#include <stdio.h>

__global__ void helloKernel(void) {
    printf("Hello from the GPU!\n");
}

int main() { 
    helloKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;

}