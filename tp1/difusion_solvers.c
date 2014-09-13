#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void explicit_method( int N, double time_interval);
void implicit_method( int N, double time_interval, double alpha);

double solve(double* actual, double* initial, int index, double time_interval, double alpha);

void print_double_vector(double* vector, int N);
double calc_difference(double* res, double* initial, int N);

#ifndef abs
#define abs(val) ((val) < 0 ? -(val) : (val))
#endif

#ifndef sqr
#define sqr(x) (x*x)
#endif


int main(int argc, char const* argv[])
{
    if (argc < 3) {
        printf("Falta el solver a usar y el time_interval\n");
        return 1;
    }
    int N = 11;
    double time_interval;
    char solver = argv[1][0];
    sscanf(argv[2], "%lf", &time_interval);

    switch (solver) {
        case 'E': explicit_method(N, time_interval);
            break;
        case 'I': implicit_method(N, time_interval, 0.5);
            break;
        case 'S': implicit_method(N, time_interval, 1);
            break;
        default: printf("Opcion invalida\n E = metodo explicito \n I = metodo implicito \n S = metodo fuertemente implicito");
    }

    return 0;
}


void print_double_vector(double* vector, int N)
{
    for (int i = 0; i < N; ++i) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}


double calc_difference(double* res, double* initial, int N)
{
    double max = 0;
    double temp;

    for (int i = 0; i < N; ++i) {
        temp = abs(res[i] - initial[i]);
        if ( temp > max) {
            max = temp;
        }
    }
    return max;
}


void explicit_method( int N, double time_interval)
{
    //setting boundary conditions
    double* initial = calloc(N, sizeof(double));
    double* res = calloc(N, sizeof(double));

    initial[0] = res[0] = 10;
    initial[N - 1] = res[N - 1] = -5;

    double threshold_convergence = 0.0001;
    double current_convergence = 10000;

    while (current_convergence > threshold_convergence) {
        print_double_vector(res, N);

        for (int i = 1; i < N - 1; ++i) {
            res[i] = solve(res, initial, i, time_interval, 0);
        }
        current_convergence = calc_difference(res, initial, N);

        memcpy(initial, res, N * sizeof(double));
    }
    free(res);
    free(initial);
}


void implicit_method(int N, double time_interval, double alpha)
{
    //setting boundary conditions
    double* initial = calloc(N, sizeof(double));
    double* res = calloc(N, sizeof(double));
    double* temp = calloc(N, sizeof(double));
    temp[0] = initial[0] = res[0] = 10;
    temp[N - 1] = initial[N - 1] = res[N - 1] = -5;

    double gs_threshold_convergence = 0.0001;
    double threshold_convergence = 0.0001;
    double current_convergence = 10000;

    while (current_convergence > threshold_convergence) {
        print_double_vector(initial, N);

        double gs_convergence = 10000;
        while (gs_convergence > gs_threshold_convergence) {
            for (int i = 1; i < N - 1; ++i) {
                res[i] = solve(res, initial, i, time_interval, 0.5);
            }

            gs_convergence = calc_difference(res, temp, N);
            memcpy(temp, res, N * sizeof(double));
        }

        current_convergence = calc_difference(res, initial, N);
        memcpy(initial, res, N * sizeof(double));
    }
    free(res);
    free(initial);
}


double solve(double* actual, double* initial, int index, double time_interval, double alpha)
{
    const double K = 0.01;
    const double int_len = 1.0 / 10.0;
    return (
               K * (
                   alpha * (time_interval / sqr(int_len)) * (actual[index + 1] + actual[index - 1])
                   + (1 - alpha) * (time_interval / sqr(int_len)) *
                   (initial[index + 1] - 2 * initial[index] + initial[index - 1])
               ) + initial[index]
           ) / (1 + (2 * K * alpha * time_interval) / sqr(int_len) );
}

