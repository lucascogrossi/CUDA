#include <stdio.h>
#include <stdlib.h>

#define SIZE 8

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    float* A, *B, *C;

    A = (float*) malloc(sizeof(float) * SIZE);
    B = (float*) malloc(sizeof(float) * SIZE);
    C = (float*) malloc(sizeof(float) * SIZE);
   
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = i;
    }

    vecAdd(A, B, C, SIZE);

    for (int i = 0; i < SIZE; i++)
        printf("%f ", C[i]);
    printf("\n");
}