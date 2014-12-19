//gcc -O3 -Wall -pedantic -fopenmp main.c
#include <omp.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <sys/time.h>

#define RNG_MOD 0x8000000000000000
uint64_t state;

uint64_t rng_int(void);
double rng_doub(double range);

int main(int argc, char const* argv[])
{
    double x, y, pi;

    if (argc < 2) {
        printf("Falta cantidad de iteraciones\n");
        return 1;
    }

    uint64_t n;// = 17179869184;

    sscanf(argv[1], "%" PRIu64, &n);

    omp_set_num_threads(4);
    uint64_t numIn = 0;

    struct timeval begin, end;
    int diff;
    do {
        gettimeofday(&begin, NULL);

        #pragma omp threadprivate(state)
        #pragma omp parallel private(x, y) reduction(+:numIn)
        {
            state = 25234 + 17 * omp_get_thread_num();
            #pragma omp for
            for (uint64_t i = 0; i <= n; i++) {
                x = (double)rng_doub(1.0);
                y = (double)rng_doub(1.0);
                if (x * x + y * y <= 1) {
                    numIn++;
                }
            }
        }
        gettimeofday(&end, NULL);

        diff = end.tv_usec - begin.tv_usec;
        // if (diff >= 0) {
        //     printf("%d\n", diff);
        // }
    }
    while (diff < 0);

    pi = (4.0 * numIn) / n ;
    printf("%f\n", pi);
    return 0;
}

uint64_t rng_int(void)
{
    // & 0x7fffffff is equivalent to modulo with RNG_MOD = 2^31
    return (state = (state * 1103515245 + 12345) & 0x7fffffffffffffff);
}

double rng_doub(double range)
{
    return ((double)rng_int()) / (((double)RNG_MOD) / range);
}

