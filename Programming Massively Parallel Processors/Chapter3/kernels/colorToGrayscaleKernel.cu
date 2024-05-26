__global __ void colorToGrayscaleKernel(unsigned char* red, unsigned char* green, unsigned char* blue, 
                                  unsigned char* gray, unsigned int width, unsigned int height) {

    // Global thread indexing 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary condition
    if (row < height && col < width) {
    
        // Linear index of pixel (row-major order)
        unsigned i = row * width + col;

        // Convert to grayscale
        gray[i] = red[i] * 0.21f + green[i] * 0.72f + blue[i] * 0.07f;
    }
}


int main() {
    // ...

    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    colorToGrayscaleKernel<<<numBlocks, numThreadsPerBlock>>>(red_d , green_d, blue_d, gray_d, width, height);
}