/*
    Parallelization approach: assign one thread to each output pixel
    and have it read multiple input pixels.
*/

int BLUR_SIZE = 1;

__global__ void blurKernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {

    // Global thread indexing for output image
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for output pixel
    if (outRow < height && outCol < width) {

        // Loop through neighbouring pixels and compute average
        int sum = 0;
        for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow) {
            for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol) {
                average += image[inRow * width + inCol];
            }
        }
        blurred[outRow * width + outCol] = sum / (unsigned char) ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1));       
    }
}

int main() {
    // ...

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    blurKernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
}