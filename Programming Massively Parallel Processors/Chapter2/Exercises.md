# Exercises

1. 
If we want to use each thread in a grid to calculate one output element of a
vector addition, what would be the expression for mapping the thread/block
indices to the data index (i)?

(A) i=threadIdx.x + threadIdx.y;  

(B) i=blockIdx.x + threadIdx.x;  

(C) i=blockIdx.x * blockDim.x + threadIdx.x;  

(D) i=blockIdx.x * threadIdx.x;  


// TODO