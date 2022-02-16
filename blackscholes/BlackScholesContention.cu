/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

// #include <helper_functions.h>  // helper functions for string parsing
// #include <helper_cuda.h>  // helper functions CUDA error checking and initialization


////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//const int OPT_N = 4000000;
const int NUM_ITERATIONS = 2048;

//const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    float *put, *call, elapsed;
    cudaMalloc((void **)&put, sizeof(float) * 1792);
    cudaMalloc((void **)&call, sizeof(float) * 1792);

    float *put2, *call2;
    cudaMalloc((void **)&put2, sizeof(float) * 1792);
    cudaMalloc((void **)&call2, sizeof(float) * 1792);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//    for (int i = 0; i < NUM_ITERATIONS; i++) {
//        BlackScholesGPU<<<14, 128, 0, stream2>>>(put2, call2, RISKFREE, VOLATILITY);
//    }

    cudaEventRecord(start, stream1);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        BlackScholesGPU<<<14, 128, 0, stream1>>>(put, call, RISKFREE, VOLATILITY);
        BlackScholesGPU<<<14, 128, 0, stream2>>>(put2, call2, RISKFREE, VOLATILITY);
    }
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("%.2f ms\n", elapsed);

    cudaFree(put);
    cudaFree(put2);
    cudaFree(call);
    cudaFree(call2);

    exit(EXIT_SUCCESS);
}

