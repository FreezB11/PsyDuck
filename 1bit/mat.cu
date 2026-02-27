// #include <cuda.h>
// #include <stdio.h>
// #include <stdint.h>

// #define N 4096
// #define WORDS (N/64)
// #define TILE 16

// typedef unsigned long long u64;

// __global__ void matmul_kernel(
//     const u64* A_val, const u64* A_sign,
//     const u64* B_val, const u64* B_sign,
//     int* C)
// {
//     __shared__ u64 sA_val[TILE][WORDS];
//     __shared__ u64 sA_sign[TILE][WORDS];

//     __shared__ u64 sB_val[TILE][WORDS];
//     __shared__ u64 sB_sign[TILE][WORDS];

//     int row = blockIdx.y * TILE + threadIdx.y;
//     int col = blockIdx.x * TILE + threadIdx.x;

//     if(row >= N || col >= N) return;

//     int sum = 0;

//     // load rows into shared memory
//     for(int k = threadIdx.x; k < WORDS; k += TILE)
//     {
//         sA_val[threadIdx.y][k] = A_val[row * WORDS + k];
//         sA_sign[threadIdx.y][k] = A_sign[row * WORDS + k];

//         sB_val[threadIdx.y][k] = B_val[col * WORDS + k];
//         sB_sign[threadIdx.y][k] = B_sign[col * WORDS + k];
//     }

//     __syncthreads();

//     // compute dot
//     #pragma unroll
//     for(int k=0;k<WORDS;k++)
//     {
//         u64 mask = sA_val[threadIdx.y][k] & sB_val[threadIdx.x][k];
//         u64 diff = sA_sign[threadIdx.y][k] ^ sB_sign[threadIdx.x][k];

//         sum += __popcll(mask & ~diff);
//         sum -= __popcll(mask & diff);
//     }

//     C[row * N + col] = sum;
// }
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#define N 4096
#define WORDS (N/64)

#define TILE 16
#define K_TILE 4   // number of u64 processed per iteration

typedef unsigned long long u64;

__global__ void matmul_kernel(
    const u64* A_val, const u64* A_sign,
    const u64* B_val, const u64* B_sign,
    int* C)
{
    __shared__ u64 sA_val[TILE][K_TILE];
    __shared__ u64 sA_sign[TILE][K_TILE];
    __shared__ u64 sB_val[TILE][K_TILE];
    __shared__ u64 sB_sign[TILE][K_TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if(row >= N || col >= N) return;

    int sum = 0;

    // iterate over K dimension in tiles
    for(int kt = 0; kt < WORDS; kt += K_TILE)
    {
        // load small tile into shared memory
        if(threadIdx.x < K_TILE)
        {
            sA_val[threadIdx.y][threadIdx.x] =
                A_val[row * WORDS + kt + threadIdx.x];

            sA_sign[threadIdx.y][threadIdx.x] =
                A_sign[row * WORDS + kt + threadIdx.x];

            sB_val[threadIdx.y][threadIdx.x] =
                B_val[col * WORDS + kt + threadIdx.x];

            sB_sign[threadIdx.y][threadIdx.x] =
                B_sign[col * WORDS + kt + threadIdx.x];
        }

        __syncthreads();

        // compute partial dot
        #pragma unroll
        for(int k = 0; k < K_TILE; k++)
        {
            u64 mask = sA_val[threadIdx.y][k] & sB_val[threadIdx.x][k];
            u64 diff = sA_sign[threadIdx.y][k] ^ sB_sign[threadIdx.x][k];

            sum += __popcll(mask & ~diff);
            sum -= __popcll(mask & diff);
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    size_t bytes = N * WORDS * sizeof(u64);

    u64 *A_val,*A_sign,*B_val,*B_sign;
    int *C;

    cudaMalloc(&A_val, bytes);
    cudaMalloc(&A_sign, bytes);
    cudaMalloc(&B_val, bytes);
    cudaMalloc(&B_sign, bytes);
    cudaMalloc(&C, N*N*sizeof(int));

    u64 *tmp = (u64*)malloc(bytes);
    for(int i=0;i<N*WORDS;i++) tmp[i] = rand();

    cudaMemcpy(A_val,tmp,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(A_sign,tmp,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_val,tmp,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_sign,tmp,bytes,cudaMemcpyHostToDevice);
    free(tmp);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;

    matmul_kernel<<<grid,block>>>(A_val,A_sign,B_val,B_sign,C);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
        matmul_kernel<<<grid,block>>>(A_val,A_sign,B_val,B_sign,C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    printf("Matrix size: %dx%d\n",N,N);
    printf("Per matmul: %.4f ms\n", ms/iters);
}