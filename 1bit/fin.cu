#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define N 8192
#define WORDS (N/64)

#define TILE 16
#define K_TILE 4

typedef unsigned long long u64;


// =====================================================
// CPU: TRANSPOSE B INTO BIT-SLICED LAYOUT
// =====================================================

void transpose_bitplanes(
    u64* src_val, u64* src_sign,
    u64* dst_val, u64* dst_sign)
{
    memset(dst_val, 0, N*WORDS*sizeof(u64));
    memset(dst_sign, 0, N*WORDS*sizeof(u64));

    for(int r=0;r<N;r++)
    for(int w=0;w<WORDS;w++)
    {
        u64 v = src_val[r*WORDS + w];
        u64 s = src_sign[r*WORDS + w];

        for(int bit=0;bit<64;bit++)
        {
            if(!(v & (1ULL<<bit)) && !(s & (1ULL<<bit))) continue;

            int c = w*64 + bit;

            if(v & (1ULL<<bit))
                dst_val[w*N + c] |= (1ULL<<r);

            if(s & (1ULL<<bit))
                dst_sign[w*N + c] |= (1ULL<<r);
        }
    }
}


// =====================================================
// CUDA KERNEL — BITNET FAST MATMUL
// =====================================================

__global__ void matmul_kernel(
    const u64* A_val, const u64* A_sign,
    const u64* B_val, const u64* B_sign,
    int* C)
{
    __shared__ u64 sA_val[TILE][K_TILE];
    __shared__ u64 sA_sign[TILE][K_TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if(row >= N || col >= N) return;

    int sum = 0;

    for(int kt=0; kt<WORDS; kt+=K_TILE)
    {
        if(threadIdx.x < K_TILE)
        {
            sA_val[threadIdx.y][threadIdx.x] =
                A_val[row*WORDS + kt + threadIdx.x];

            sA_sign[threadIdx.y][threadIdx.x] =
                A_sign[row*WORDS + kt + threadIdx.x];
        }

        __syncthreads();

        #pragma unroll
        for(int k=0;k<K_TILE;k++)
        {
            u64 a_val = sA_val[threadIdx.y][k];
            u64 a_sign = sA_sign[threadIdx.y][k];

            u64 b_val = B_val[(kt+k)*N + col];
            u64 b_sign = B_sign[(kt+k)*N + col];

            u64 mask = a_val & b_val;
            u64 diff = a_sign ^ b_sign;

            sum += __popcll(mask & ~diff);
            sum -= __popcll(mask & diff);
        }

        __syncthreads();
    }

    C[row*N + col] = sum;
}


// =====================================================
// MAIN BENCHMARK
// =====================================================

int main()
{
    size_t bytes = N*WORDS*sizeof(u64);

    u64 *A_val = (u64*)malloc(bytes);
    u64 *A_sign = (u64*)malloc(bytes);

    u64 *B_val = (u64*)malloc(bytes);
    u64 *B_sign = (u64*)malloc(bytes);

    u64 *BvT = (u64*)malloc(bytes);
    u64 *BsT = (u64*)malloc(bytes);

    // random init
    for(int i=0;i<N*WORDS;i++)
    {
        A_val[i] = rand();
        A_sign[i] = rand();

        B_val[i] = rand();
        B_sign[i] = rand();
    }

    printf("Transposing B bitplanes...\n");
    transpose_bitplanes(B_val, B_sign, BvT, BsT);

    u64 *dA_val,*dA_sign,*dB_val,*dB_sign;
    int *dC;

    cudaMalloc(&dA_val, bytes);
    cudaMalloc(&dA_sign, bytes);
    cudaMalloc(&dB_val, bytes);
    cudaMalloc(&dB_sign, bytes);
    cudaMalloc(&dC, N*N*sizeof(int));

    cudaMemcpy(dA_val,A_val,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dA_sign,A_sign,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB_val,BvT,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB_sign,BsT,bytes,cudaMemcpyHostToDevice);

    dim3 block(TILE,TILE);
    dim3 grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 50;

    matmul_kernel<<<grid,block>>>(dA_val,dA_sign,dB_val,dB_sign,dC);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
        matmul_kernel<<<grid,block>>>(dA_val,dA_sign,dB_val,dB_sign,dC);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    printf("\n===== RESULTS =====\n");
    printf("Matrix: %dx%d\n",N,N);
    printf("Per matmul: %.3f ms\n", ms/iters);

    cudaFree(dA_val);
    cudaFree(dA_sign);
    cudaFree(dB_val);
    cudaFree(dB_sign);
    cudaFree(dC);

    free(A_val); free(A_sign);
    free(B_val); free(B_sign);
    free(BvT); free(BsT);
}