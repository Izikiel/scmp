#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>


int main(int argc, char const* argv[])
{
    if (argc < 2) {
        printf("Falta cantidad de iteraciones luego de warmup\n");
        return 1;
    }
    int after_n;
    sscanf(argv[1], "%d", &after_n);

    double R = 4;
    int thread_num = 1000;
    int top_n = 10000;
    double step = (R / ((double)thread_num));

    double solutions [thread_num][after_n];
    // struct timeval begin, end;
    // gettimeofday(&begin, NULL);

    for (int id = 0; id < thread_num; ++id) {
        double lambda = id * step;
        double X = 0.5;

        for (int iter = 0; iter < top_n; iter++) {
            X = lambda * X * (1 - X);
        }

        for (int iter = 0; iter < after_n; ++iter) {
            X = lambda * X * (1 - X);
            solutions[id][iter] = X;
        }
    }

    // gettimeofday(&end, NULL);
    // printf("%d\n", (int) (end.tv_usec - begin.tv_usec));

    int k = 0;
    for (double i = step ; i <= R; i += step, k++) {
        for (int j = 0; j < after_n; ++j) {
            printf("%f %f\n", i, solutions[k][j]);
        }
    }


    return 0;
}
