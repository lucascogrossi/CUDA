Exercises

1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

    (A) i=threadIdx.x + threadIdx.y;

    (B) i=blockIdx.x + threadIdx.x;

    (C) i=blockIdx.x * blockDim.x + threadIdx.x;

    (D) i=blockIdx.x * threadIdx.x;

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/blockindices to the data index (i) of the first element to be processed by a thread?

    (A) i=blockIdx.x * blockDim.x + threadIdx.x +2;

    (B) i=blockIdx.x * threadIdx.x * 2;

    (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;

    (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2 * blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for apping the thread/block indices to data index of the first element?

    (A) i=blockIdx.x * blockDim.x + threadIdx.x + 2;

    (B) i=blockIdx.x * threadIdx.x * 2;

    (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;

    (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

    (A) 8000 

    (B) 8196

    (C) 8192

    (D) 8200
    
5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?

    (A) n

    (B) v

    (C) n * sizeof(int)

    (D) v * sizeof(int)

6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc
() call?


    (A) n

    (B) (void* ) A_d

    (C) *A_d

    (D) (void** ) &A_d
