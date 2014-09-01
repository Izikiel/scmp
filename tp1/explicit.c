#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double solve_explicit(double next, double actual, double prev, double time_interval);
void explicit_method(double *res, double *initial, int N, double time_interval);

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
    if (argc < 2) {
        printf("Falta el time_interval\n");
        return 1;
    }
    int N = 11;
    double time_interval;
    sscanf(argv[1], "%lf", &time_interval);
    double *initial = calloc(N, sizeof(double));
    double *res = calloc(N, sizeof(double));

    explicit_method(res, initial, N, time_interval);

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
    const double K = 0.01; //??? Adimensionalizar
    const double int_len = 1.0 / 10.0;
    double res = (actual + time_interval * K  * ((next + prev - 2 * actual) / sqr(int_len)));
    return res;
}



