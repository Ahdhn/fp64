// code taken from
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/coalescing-global/coalescing.cu
/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gtest/gtest.h"
#include "helper.h"


template <typename T>
__global__ void offset(T* a, int s, bool with_division, T division_factor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + s;
    T   p = 1;
    if (with_division) {
        p = p / (T(i) * division_factor);
    }
    a[i] = a[i] + p;
}


template <typename T>
__global__ void stride(T* a, int s, bool with_division, T division_factor)
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
    T   p = 1;
    if (with_division) {
        p = p / (T(i) * division_factor);
    }
    a[i] = a[i] + p;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
    int blockSize = 256;

    T*          d_a;
    cudaEvent_t startEvent, stopEvent;

    int n = nMB * 1024 * 1024 / sizeof(T);

    // NB:  d_a(33*nMB) for stride case
    CUDA_ERROR(cudaMalloc(&d_a, n * 33 * sizeof(T)));

    CUDA_ERROR(cudaEventCreate(&startEvent));
    CUDA_ERROR(cudaEventCreate(&stopEvent));


    srand(time(NULL));
    T divison_factor = T(rand()) / (T(RAND_MAX) + 1.0);

    const char separator = ' ';
    std::cout << std::endl;

    auto print_header = [&]() {
        std::cout << std::left << std::setw(8) << std::setfill(separator)
                  << "Stride";
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << "BW (GB/s)";
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << "Time (ms)";
        std::cout << std::left << std::setw(20) << std::setfill(separator)
                  << "BW w/div (GB/s)";
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << "Time w/div (ms)" << std::endl;
    };

    std::cout << "============================== Stride Test "
                 "=============================="
              << std::endl;
    print_header();

    stride<<<n / blockSize, blockSize>>>(d_a, 1, false,
                                         divison_factor);  // warm up

    for (int i = 1; i <= 32; i++) {
        CUDA_ERROR(cudaMemset(d_a, 0, n * sizeof(T)));

        float ms, ms_div;
        CUDA_ERROR(cudaEventRecord(startEvent, 0));
        stride<<<n / blockSize, blockSize>>>(d_a, i, false, divison_factor);
        CUDA_ERROR(cudaEventRecord(stopEvent, 0));
        CUDA_ERROR(cudaEventSynchronize(stopEvent));
        CUDA_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));


        CUDA_ERROR(cudaEventRecord(startEvent, 0));
        stride<<<n / blockSize, blockSize>>>(d_a, i, true, divison_factor);
        CUDA_ERROR(cudaEventRecord(stopEvent, 0));
        CUDA_ERROR(cudaEventSynchronize(stopEvent));
        CUDA_ERROR(cudaEventElapsedTime(&ms_div, startEvent, stopEvent));


        std::cout << std::left << std::setw(8) << std::setfill(separator) << i;
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ((2 * nMB) / ms);
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ms;
        std::cout << std::left << std::setw(20) << std::setfill(separator)
                  << ((2 * nMB) / ms_div);
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ms_div << std::endl;
    }

    std::cout << "\n============================== Offset Test "
                 "=============================="
              << std::endl;
    print_header();
    for (int i = 0; i <= 32; i++) {
        CUDA_ERROR(cudaMemset(d_a, 0, n * sizeof(T)));

        float ms, ms_div;
        CUDA_ERROR(cudaEventRecord(startEvent, 0));
        offset<<<n / blockSize, blockSize>>>(d_a, i, false, divison_factor);
        CUDA_ERROR(cudaEventRecord(stopEvent, 0));
        CUDA_ERROR(cudaEventSynchronize(stopEvent));
        CUDA_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));


        CUDA_ERROR(cudaEventRecord(startEvent, 0));
        offset<<<n / blockSize, blockSize>>>(d_a, i, true, divison_factor);
        CUDA_ERROR(cudaEventRecord(stopEvent, 0));
        CUDA_ERROR(cudaEventSynchronize(stopEvent));
        CUDA_ERROR(cudaEventElapsedTime(&ms_div, startEvent, stopEvent));


        std::cout << std::left << std::setw(8) << std::setfill(separator) << i;
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ((2 * nMB) / ms);
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ms;
        std::cout << std::left << std::setw(20) << std::setfill(separator)
                  << ((2 * nMB) / ms_div);
        std::cout << std::left << std::setw(15) << std::setfill(separator)
                  << ms_div << std::endl;
    }

    CUDA_ERROR(cudaEventDestroy(startEvent));
    CUDA_ERROR(cudaEventDestroy(stopEvent));
    CUDA_ERROR(cudaFree(d_a));
}
TEST(Test, simple)
{
    int  nMB = 32;
    int  deviceId = 0;
    bool bFp64 = true;

    cudaDeviceProp prop;

    CUDA_ERROR(cudaSetDevice(deviceId));

    CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    printf("Device: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", nMB);
    printf("%s Precision\n", bFp64 ? "Double" : "Single");

    if (bFp64) {
        runTest<double>(deviceId, nMB);
    } else {
        runTest<float>(deviceId, nMB);
    }

    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
