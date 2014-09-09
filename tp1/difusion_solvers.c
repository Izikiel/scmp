#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double solve_explicit(double next, double actual, double prev, double time_interval);
void explicit_method(double *res, double *initial, int N, double time_interval);

void implicit_method(double *res, double *initial, int N, double time_interval);
double solve_implicit(double *actual, double *initial, int index, double time_interval);

void print_double_vector(double *vector, int N);
double calc_difference(double *res, double *initial, int N);

#ifndef abs
#define abs(val) ((val) < 0 ? -(val) : (val))
#endif

#ifndef sqr
#define sqr(x) (x*x)
#endif


int main(int argc, char const *argv[])
{
    if (argc < 3) {
        printf("Falta el solver a usar y el time_interval\n");
        return 1;
    }
    int N = 11;
    double time_interval;
    char solver = argv[1][0];
    sscanf(argv[2], "%lf", &time_interval);
    double *initial = calloc(N, sizeof(double));
    double *res = calloc(N, sizeof(double));


    if (solver == 'E') {
        explicit_method(res, initial, N, time_interval);
    } else {
        implicit_method(res, initial, N, time_interval);
    }

    free(res);
    free(initial);

    return 0;
}


void print_double_vector(double *vector, int N)
{
    for (int i = 0; i < N; ++i) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}


double calc_difference(double *res, double *initial, int N)
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

void explicit_method(double *res, double *initial, int N, double time_interval)
{
    //setting boundary conditions

    initial[0] = res[0] = 10;
    initial[N - 1] = res[N - 1] = -5;

    double threshold_convergence = 0.0001;
    double current_convergence = 10000;

    while (current_convergence > threshold_convergence) {
        print_double_vector(res, N);

        for (int i = 1; i < N - 1; ++i) {
            res[i] = solve_explicit(initial[i + 1], initial[i], initial[i - 1], time_interval);
        }
        current_convergence = calc_difference(res, initial, N);

        memcpy(initial, res, N * sizeof(double));
    }
}

double solve_explicit(double next, double actual, double prev, double time_interval)
{
    const double K = 0.01;
    const double int_len = 1.0 / 10.0;
    double res = (actual + time_interval * K  * ((next + prev - 2 * actual) / sqr(int_len)));
    return res;
}



void implicit_method(double *res, double *initial, int N, double time_interval)
{
    //setting boundary conditions
    double *temp = calloc(N, sizeof(double));
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
                res[i] = solve_implicit(res, initial, i, time_interval);
            }

            gs_convergence = calc_difference(res, temp, N);

            memcpy(temp, res, N * sizeof(double));
        }

        current_convergence = calc_difference(res, initial, N);

        memcpy(initial, res, N * sizeof(double));
    }
}

double solve_implicit(double *actual, double *initial, int index, double time_interval)
{
    const double K = 0.01;
    const double int_len = 1.0 / 10.0;
    double alpha = 0.5;
    double res = (
                K * (
                        alpha * (time_interval/sqr(int_len)) * (actual[index+1]+actual[index-1]) 
                        + (1-alpha) * (time_interval/sqr(int_len)) * (initial[index+1] - 2 * initial[index] + initial[index-1])
                        + initial[index]
                    )
                )/ (1 + (2* K * alpha * time_interval)/sqr(int_len) );

    return res;
}


