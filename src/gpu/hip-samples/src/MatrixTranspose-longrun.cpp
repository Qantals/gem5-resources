/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include<iostream>

// hip header file
#include "hip/hip_runtime.h"

#include <stdint.h>
#include <gem5/m5ops.h>

#define WIDTH     1024


#define NUM       (WIDTH*WIDTH)

#define THREADS_PER_BLOCK_X  4
#define THREADS_PER_BLOCK_Y  4
#define THREADS_PER_BLOCK_Z  1

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void matrixTranspose(float *out,
                                const float *in,
                                const int width,
                                const int inner_iters)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= width || y >= width) return;

    size_t idx_in = (size_t)x * width + y;
    float val = in[idx_in];
    for (int it = 0; it < inner_iters; ++it) {
        val = val * 1.0000001f + 0.000001f;
    }
    out[y * width + x] = val;
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(
    float * output,
    float * input,
    const unsigned int width)
{
    for(unsigned int j=0; j < width; j++)
    {
        for(unsigned int i=0; i < width; i++)
        {
            output[i*width + j] = input[j*width + i];
        }
    }
}

int main(int argc, char** argv) {

    float* Matrix = nullptr;
    float* gpuTransposeMatrix = nullptr;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int kernel_repeats = 100;
    int inner_iters = 256;
    if (argc >= 2) kernel_repeats = atoi(argv[1]);
    if (argc >= 3) inner_iters = atoi(argv[2]);

    printf("info: WIDTH=%d NUM=%d kernel_repeats=%d inner_iters=%d\n", WIDTH, NUM, kernel_repeats, inner_iters);

    // allocate host-pinned input and output so kernel can read/write directly
    hipHostMalloc(&Matrix, NUM * sizeof(float));
    hipHostMalloc(&gpuTransposeMatrix, NUM * sizeof(float));

    // initialize the input data on CPU (kept, but WIDTH reduced to limit overhead)
    for (int i = 0; i < NUM; i++) {
        Matrix[i] = (float)i*10.0f;
    }

    const unsigned blocksX = WIDTH/THREADS_PER_BLOCK_X;
    const unsigned blocksY = WIDTH/THREADS_PER_BLOCK_Y;

    m5_work_begin(0, 0);

    printf("info: launch 'matrixTranspose' kernel\n");

    for (int rep = 0; rep < kernel_repeats; rep++) {
        hipLaunchKernelGGL(matrixTranspose,
                                             dim3(blocksX, blocksY),
                                             dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                                             0, 0,
                                             gpuTransposeMatrix, Matrix, WIDTH, inner_iters);
    }
    hipDeviceSynchronize();

    m5_work_end(0, 0);

    // sample an element to ensure kernel executed
    float sample = gpuTransposeMatrix[0];
    printf("info: sample output = %f\n", sample);

    // free the resources
    hipHostFree(gpuTransposeMatrix);
    hipHostFree(Matrix);

    return 0;
}
