// We assume that the program has a defined constant BLUR_SIZE.
//The value of BLUR_SIZE is set such that BLUR_SIZE gives the number of pixels on
//each side (radius) of the patch and 2 * BLUR_SIZE+1 gives the total number of pixels
//across one dimension of the patch

int BLUR_SIZE = 1;

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curRol];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }

        }
        out[row * w  + col] = (unsigned char) (pixelVal / pixels);
    }
}