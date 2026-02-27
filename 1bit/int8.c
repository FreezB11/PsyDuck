#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <linux/time.h>


#define N 256   // must be multiple of 256

typedef uint64_t u64;

u64 A_val[N][N/64];
u64 A_sign[N][N/64];

u64 B_val[N][N/64];
u64 B_sign[N][N/64];

int C[N][N];


// AVX2 popcount via lookup trick
static inline int popcnt256(__m256i v)
{
    uint64_t tmp[4];
    _mm256_storeu_si256((__m256i*)tmp, v);
    return __builtin_popcountll(tmp[0])
         + __builtin_popcountll(tmp[1])
         + __builtin_popcountll(tmp[2])
         + __builtin_popcountll(tmp[3]);
}


int ternary_dot(int i, int j)
{
    int pos = 0, neg = 0;

    for (int k = 0; k < N/64; k += 4)
    {
        __m256i av = _mm256_loadu_si256((__m256i*)&A_val[i][k]);
        __m256i as = _mm256_loadu_si256((__m256i*)&A_sign[i][k]);
        __m256i bv = _mm256_loadu_si256((__m256i*)&B_val[j][k]);
        __m256i bs = _mm256_loadu_si256((__m256i*)&B_sign[j][k]);

        __m256i mask = _mm256_and_si256(av, bv);
        __m256i diff = _mm256_xor_si256(as, bs);

        __m256i same = _mm256_andnot_si256(diff, mask);
        __m256i opposite = _mm256_and_si256(diff, mask);

        pos += popcnt256(same);
        neg += popcnt256(opposite);
    }

    return pos - neg;
}


void matmul()
{
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            C[i][j] = ternary_dot(i,j);
}


long now_ns()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e9 + t.tv_nsec;
}


int main()
{
    srand(0);

    // random fill
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    {
        int v = rand()%3 -1;
        if(v) A_val[i][j/64] |= 1ULL<<(j%64);
        if(v<0) A_sign[i][j/64] |= 1ULL<<(j%64);

        v = rand()%3 -1;
        if(v) B_val[j][i/64] |= 1ULL<<(i%64);
        if(v<0) B_sign[j][i/64] |= 1ULL<<(i%64);
    }

    long s = now_ns();
    matmul();
    long e = now_ns();

    printf("Time: %.2f ms\n", (e-s)/1e6);
}