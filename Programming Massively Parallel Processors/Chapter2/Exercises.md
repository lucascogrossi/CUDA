# Exercises

1. If we want to use each thread in a grid to calculate one output element of a
vector addition, what would be the expression for mapping the thread/block
indices to the data index (i)?

(A) i=threadIdx.x + threadIdx.y;  
(B) i=blockIdx.x + threadIdx.x;  
(C) i=blockIdx.x * blockDim.x + threadIdx.x;  
(D) i=blockIdx.x * threadIdx.x;  

2. Assume that we want to use each thread to calculate two adjacent elements of
a vector addition. What would be the expression for mapping the thread/block
indices to the data index (i) of the first element to be processed by a thread?

(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;  
(B) i=blockIdx.x * threadIdx.x * 2;  
(C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;  
(D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;  

3.
We want to use each thread to calculate two elements of a vector addition.
Each thread block processes 2 * blockDim.x consecutive elements that form
two sections. All threads in each block will process a section first, each
processing one element. They will then all move to the next section, eachExercises
processing one element. Assume that variable i should be the index for the
first element to be processed by a thread. What would be the expression for
mapping the thread/block indices to data index of the first element?

(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;
(B) i=blockIdx.x * threadIdx.x * 2;
(C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
(D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;





// TODO