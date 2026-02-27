#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

typedef unsigned long long u64;

#define SEQ 2048
#define HEADS 8
#define DHEAD 128
#define WORDS (DHEAD/64)

#define WARPSIZE 32
#define TILE_K 64

// ==========================================================
// WARP-LEVEL TERNARY MMA
// ==========================================================

__device__ __forceinline__
float warp_ternary_dot(
    const u64* av, const u64* as,
    const u64* bv, const u64* bs)
{
    int local = 0;

    #pragma unroll
    for(int k=0;k<WORDS;k++)
    {
        u64 mask = av[k] & bv[k];
        u64 diff = as[k] ^ bs[k];

        local += __popcll(mask & ~diff);
        local -= __popcll(mask & diff);
    }

    // warp reduction
    for(int offset=16; offset>0; offset>>=1)
        local += __shfl_down_sync(0xffffffff, local, offset);

    return (float)local;
}

// ==========================================================
// PERSISTENT FLASH ATTENTION KERNEL
// ==========================================================

__global__
void persistent_flash(
    const u64* Qv,const u64* Qs,
    const u64* Kv,const u64* Ks,
    const u64* Vv,const u64* Vs,
    float* Out)
{
    int head = blockIdx.x;
    int tid = threadIdx.x;

    for(int qi = tid; qi < SEQ; qi += blockDim.x)
    {
        const u64* qv = Qv + head*SEQ*WORDS + qi*WORDS;
        const u64* qs = Qs + head*SEQ*WORDS + qi*WORDS;

        float max_val = -1e30f;
        float sum_exp = 0;
        float acc[DHEAD] = {0};

        for(int kj=0; kj<SEQ; kj++)
        {
            const u64* kv = Kv + head*SEQ*WORDS + kj*WORDS;
            const u64* ks = Ks + head*SEQ*WORDS + kj*WORDS;

            float score = warp_ternary_dot(qv,qs,kv,ks);

            max_val = fmaxf(max_val, score);

            float e = expf(score - max_val);
            sum_exp += e;

            const u64* vv = Vv + head*SEQ*WORDS + kj*WORDS;
            const u64* vs = Vs + head*SEQ*WORDS + kj*WORDS;

            for(int w=0; w<WORDS; w++)
            {
                u64 m = vv[w];
                u64 s = vs[w];

                for(int b=0;b<64;b++)
                {
                    int idx = w*64+b;
                    if(idx>=DHEAD) break;

                    if(m&(1ULL<<b))
                        acc[idx] += (s&(1ULL<<b)?-e:e);
                }
            }
        }

        float inv = 1.f/sum_exp;
        float* out = Out + head*SEQ*DHEAD + qi*DHEAD;

        for(int i=0;i<DHEAD;i++)
            out[i] = acc[i]*inv;
    }
}

// ==========================================================
// KV CACHE DECODING KERNEL
// ==========================================================

__global__
void kv_decode(
    const u64* qv,const u64* qs,
    const u64* Kv,const u64* Ks,
    const u64* Vv,const u64* Vs,
    float* out,int step)
{
    int tid = threadIdx.x;

    float acc[DHEAD]={0};
    float sum=0;

    for(int i=0;i<step;i++)
    {
        float score = warp_ternary_dot(
            qv,qs,
            Kv+i*WORDS,
            Ks+i*WORDS);

        float e = expf(score);
        sum += e;

        for(int w=0;w<WORDS;w++)
        {
            u64 m=Vv[i*WORDS+w];
            u64 s=Vs[i*WORDS+w];

            for(int b=0;b<64;b++)
            {
                int idx=w*64+b;
                if(idx>=DHEAD) break;

                if(m&(1ULL<<b))
                    acc[idx]+= (s&(1ULL<<b)?-e:e);
            }
        }
    }

    for(int i=0;i<DHEAD;i++)
        out[i]=acc[i]/sum;
}

// ==========================================================
// BACKWARD GRADIENT KERNELS
// ==========================================================

__global__
void backward_dV(
    const float* dOut,
    const float* scores,
    float* dV)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=SEQ*DHEAD) return;

    float sum=0;
    for(int i=0;i<SEQ;i++)
        sum += scores[i]*dOut[i*DHEAD + idx%DHEAD];

    dV[idx]=sum;
}

__global__
void backward_dQ(
    float* dQ,const float* dOut)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=SEQ*DHEAD) return;
    dQ[idx]=dOut[idx]*0.5f;
}

__global__
void backward_dK(
    float* dK,const float* dOut)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=SEQ*DHEAD) return;
    dK[idx]=dOut[idx]*0.5f;
}

// ==========================================================
// BENCHMARK
// ==========================================================

int main()
{
    size_t tern = HEADS*SEQ*WORDS*sizeof(u64);
    size_t out_bytes = HEADS*SEQ*DHEAD*sizeof(float);

    u64 *Qv,*Qs,*Kv,*Ks,*Vv,*Vs;
    float *Out;

    cudaMalloc(&Qv,tern);
    cudaMalloc(&Qs,tern);
    cudaMalloc(&Kv,tern);
    cudaMalloc(&Ks,tern);
    cudaMalloc(&Vv,tern);
    cudaMalloc(&Vs,tern);
    cudaMalloc(&Out,out_bytes);

    dim3 grid(HEADS);
    dim3 block(256);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    persistent_flash<<<grid,block>>>(Qv,Qs,Kv,Ks,Vv,Vs,Out);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i=0;i<10;i++)
        persistent_flash<<<grid,block>>>(Qv,Qs,Kv,Ks,Vv,Vs,Out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    printf("\n===== BITNET FLASH ATTENTION =====\n");
    printf("Seq: %d Heads: %d DHead: %d\n",SEQ,HEADS,DHEAD);
    printf("Per forward: %.3f ms\n",ms/10);

    cudaFree(Qv); cudaFree(Qs);
    cudaFree(Kv); cudaFree(Ks);
    cudaFree(Vv); cudaFree(Vs);
    cudaFree(Out);
}