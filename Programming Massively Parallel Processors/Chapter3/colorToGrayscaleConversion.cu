int CHANNELS = 3; // We assume that CHANNELS is a constant of value 3, 
                  //and its definition is outside the kernel function.

// Each input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)

__global__ void colorToGrayscaleConvertionKernel(unsigned char* grayImage, unsigned char* rgbImage, int width, int height) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row * width + col;
        // One can think of the RGB image having CHANNEL
        // times more columns than the grayscale image
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = rgbImage[rgbOffset    ]; // Red value
        unsigned char g = rgbImage[rgbOffset + 1]; // Green value
        unsigned char b = rgbImage[rgbOffset + 2]; // Blue value
        // Perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f * r + 0.71f + 0.07f * b;
    }
}