#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double solve_explicit(double next, double actual, double prev);
double solve_implicit(double new_next, double new_prev, double next, double actual, double prev);

void print_double_vector(double* vector, int N);
double calc_difference(double* res, double* initial, int N);


#ifndef abs
#define abs(val) ((val) < 0 ? -(val) : (val))
#endif


int main(int argc, char const* argv[])
{
    int N = 11;
    double* initial = calloc(N, sizeof(double));
    double* res = calloc(N, sizeof(double));

    //setting boundary conditions

    initial[0] = res[0] = 10;
    initial[N - 1] = res[N - 1] = -5;

    double delta_convergence = 0.1;
    double current_convergence = 10000;

    while (current_convergence > delta_convergence) {
        print_double_vector(res, N);

        for (int i = 1; i < N - 1; ++i) {
            res[i] = solve_explicit(initial[i + 1], initial[i], initial[i - 1]);
        }
        current_convergence = calc_difference(res, initial, N);

        memcpy(initial, res, N * sizeof(double));
    }

    free(res);
    free(initial);

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
    //REVISAR!
    double max = 0;
    double temp;

    for (int i = 0; i < N; ++i) {
        temp = abs(res[i] - initial[i]);
        if ( temp > max) {
            max = temp;
        }
    }
    return temp;

}

double solve_explicit(double next, double actual, double prev)
{
    return -1;
}



