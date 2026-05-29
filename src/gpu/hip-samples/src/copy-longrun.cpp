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

#include <iostream>
#include <cstdlib>

// hip header file
#include "hip/hip_runtime.h"

#include <stdint.h>
#include <gem5/m5ops.h>

#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void streamCopy(float *out,
                           const float *in,
                           const size_t num,
                           const int passes)
{
    size_t idx = (size_t)hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (idx >= num) return;

    float val = 0.0f;
    for (int it = 0; it < passes; ++it) {
        val += in[idx];
    }
    out[idx] = val;
}

int main(int argc, char** argv) {

    float** hostIn = nullptr;
    float** hostOut = nullptr;
    float** deviceIn = nullptr;
    float** deviceOut = nullptr;
    hipStream_t* streams = nullptr;

    hipDeviceProp_t devProp;
    hipError_t err = hipSuccess;
    err = hipGetDeviceProperties(&devProp, 0);
    if (err != hipSuccess) {
        printf("ERROR: hipGetDeviceProperties => %d\n", err);
        return -1;
    }

    std::cout << "Device name " << devProp.name << std::endl;

    size_t bytes = 64ULL * 1024ULL * 1024ULL; // 67,108,864
    int repeats = 4;
    int num_streams = 2;
    int passes = 1;
    if (argc >= 2) bytes = strtoull(argv[1], nullptr, 10);
    if (argc >= 3) repeats = atoi(argv[2]);
    if (argc >= 4) num_streams = atoi(argv[3]);
    if (argc >= 5) passes = atoi(argv[4]);

    if (num_streams <= 0) {
        fprintf(stderr, "ERROR: num_streams must be > 0\n");
        return -1;
    }
    if (passes <= 0) {
        fprintf(stderr, "ERROR: passes must be > 0\n");
        return -1;
    }
    if (bytes < sizeof(float)) {
        fprintf(stderr, "ERROR: bytes must be >= %zu\n", sizeof(float));
        return -1;
    }
    bytes = (bytes / sizeof(float)) * sizeof(float);

    const size_t num = bytes / sizeof(float);
    printf("info: BYTES=%zu NUM=%zu repeats=%d streams=%d passes=%d\n",
           bytes, num, repeats, num_streams, passes);

    hostIn = new float*[num_streams];
    hostOut = new float*[num_streams];
    deviceIn = new float*[num_streams];
    deviceOut = new float*[num_streams];
    streams = new hipStream_t[num_streams];

    for (int s = 0; s < num_streams; ++s) {
        err = hipStreamCreate(&streams[s]);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipStreamCreate => %d\n", err);
            return -1;
        }
        err = hipHostMalloc(&hostIn[s], bytes);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipHostMalloc hostIn (size:%zu) => %d\n", bytes, err);
            return -1;
        }
        err = hipHostMalloc(&hostOut[s], bytes);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipHostMalloc hostOut (size:%zu) => %d\n", bytes, err);
            return -1;
        }
        err = hipMalloc(&deviceIn[s], bytes);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipMalloc deviceIn (size:%zu) => %d\n", bytes, err);
            return -1;
        }
        err = hipMalloc(&deviceOut[s], bytes);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipMalloc deviceOut (size:%zu) => %d\n", bytes, err);
            return -1;
        }

        for (size_t i = 0; i < num; ++i) {
            hostIn[s][i] = (float)i * 0.25f;
        }
    }

    const unsigned blocksX = (num + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;

    printf("info: launch 'streamCopy' kernel\n");

    m5_work_begin(0, 0);

    for (int rep = 0; rep < repeats; ++rep) {
        for (int s = 0; s < num_streams; ++s) {
            err = hipMemcpyAsync(deviceIn[s], hostIn[s], bytes, hipMemcpyHostToDevice, streams[s]);
            if (err != hipSuccess) {
                fprintf(stderr, "ERROR: hipMemcpyAsync H2D (size:%zu) => %d\n", bytes, err);
                return -1;
            }
            hipLaunchKernelGGL(streamCopy,
                               dim3(blocksX, 1, 1),
                               dim3(THREADS_PER_BLOCK_X, 1, 1),
                               0, streams[s],
                               deviceOut[s], deviceIn[s], num, passes);
            err = hipMemcpyAsync(hostOut[s], deviceOut[s], bytes, hipMemcpyDeviceToHost, streams[s]);
            if (err != hipSuccess) {
                fprintf(stderr, "ERROR: hipMemcpyAsync D2H (size:%zu) => %d\n", bytes, err);
                return -1;
            }
        }
    }

    for (int s = 0; s < num_streams; ++s) {
        err = hipStreamSynchronize(streams[s]);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipStreamSynchronize => %d\n", err);
            return -1;
        }
    }

    m5_work_end(0, 0);

    float sample = hostOut[0][0];
    printf("info: sample output = %f\n", sample);

    for (int s = 0; s < num_streams; ++s) {
        hipFree(deviceOut[s]);
        hipFree(deviceIn[s]);
        hipHostFree(hostOut[s]);
        hipHostFree(hostIn[s]);
        hipStreamDestroy(streams[s]);
    }

    delete[] streams;
    delete[] deviceOut;
    delete[] deviceIn;
    delete[] hostOut;
    delete[] hostIn;

    return 0;
}
