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

#include <stdio.h>
#include "hip/hip_runtime.h"

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

/*
 * Generate data and do repeated math directly on GPU to keep compute units busy.
 * Kernel writes directly into host-pinned memory allocated with hipHostMalloc,
 * which reduces explicit host-side initialization overhead.
 */
__global__ void
vector_square_longrun(float *C_h, size_t N, int inner_iters)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        float x = 1.618f + (float)i;
        for (int it = 0; it < inner_iters; it++) {
            x = x * 1.0000001f + 0.000001f;
        }
        C_h[i] = x * x;
    }
}


int main(int argc, char *argv[])
{
    size_t N = 10000000;
    int kernel_repeats = 200;
    int inner_iters = 256;
    size_t Nbytes = N * sizeof(float);

    if (argc >= 2) {
        N = strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        kernel_repeats = atoi(argv[2]);
    }
    if (argc >= 4) {
        inner_iters = atoi(argv[3]);
    }

    if (N == 0 || kernel_repeats <= 0 || inner_iters < 0) {
        fprintf(stderr, "Usage: %s [N>0] [kernel_repeats>0] [inner_iters>=0]\n", argv[0]);
        return 1;
    }

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    #ifdef __HIP_PLATFORM_HCC__
      printf ("info: architecture on AMD GPU device is: %d\n",props.gcnArch);
    #endif
    printf ("info: allocate host-pinned mem (%6.2f MB)\n", Nbytes/1024.0/1024.0);

    float *C_h = NULL;
    CHECK(hipHostMalloc(&C_h, Nbytes));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf ("info: launch 'vector_square_longrun' kernel\n");
    printf ("info: N=%zu kernel_repeats=%d inner_iters=%d\n", N, kernel_repeats, inner_iters);

    for (int rep = 0; rep < kernel_repeats; rep++) {
        hipLaunchKernelGGL(
            vector_square_longrun,
            dim3(blocks),
            dim3(threadsPerBlock),
            0,
            0,
            C_h,
            N,
            inner_iters);
    }
    CHECK(hipDeviceSynchronize());

    float sample = C_h[N - 1];
    printf("info: sample output = %f\n", sample);

    CHECK(hipHostFree(C_h));

    return 0;
}
