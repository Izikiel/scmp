#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"


#ifndef sqr
#define sqr(x) (x*x)
#endif

double solve(double *actual, double *initial, int index, double time_interval, double alpha);
void explicit_method( int N, double time_interval, double left, double right, double total_time);
double calc_difference(double *res, double *initial, int N);
void print_double_vector(double *vector, int N, int start, int stop);

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        printf("Falta el intervalo de tiempo\n");
        return -1;
    }


    double time_interval;
    sscanf(argv[1], "%lf", &time_interval);

    int id;
    int numproc;
    int source;
    int dest;

    int N = 11;

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    /* number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    MPI_Barrier(MPI_COMM_WORLD);


    if (id == 0) {
        explicit_method(N, time_interval, 1, 0.1, 100);
    } else if (id == 1) {
        explicit_method(N, time_interval, 0.1, 0, 100);
    }

    MPI_Barrier(MPI_COMM_WORLD);


    MPI_Finalize();

    return 0;
}


void explicit_method( int N, double time_interval, double left, double right, double total_time)
{
    //setting boundary conditions
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int dest = id ? 0 : 1;

    int tag = 0;
    MPI_Status status;


    double *initial = calloc(N, sizeof(double));
    double *res = calloc(N, sizeof(double));
    if (id) {
        initial[1] = res[1] = left;
        initial[N - 1] = res[N - 1] = right;
    } else {
        initial[0] = res[0] = left;
        initial[N - 2] = res[N - 2] = right;
    }


    double threshold_convergence = 0.0001;
    double current_convergence = 10000;

    const double K = 0.1;
    const double int_len = 1.0 / 20.0;

    double actual_time = 0.0;

    while (actual_time < total_time) {
        char flag;
        if (id) {
            printf("Proceso %d\n", id);
            print_double_vector(res, N, 1, N);
            printf("\n############################\n");
            MPI_Send(&flag, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&flag, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &status);
            printf("Proceso %d\n", id);
            print_double_vector(res, N, 0, N - 1);
            printf("\n############################\n");
        }

        for (int i = id ? 2 : 1; i < (N - (id ? 1 : 2)); ++i) {
            res[i] = solve(res, initial, i, time_interval, 0);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        //Meter mpi aca!

        double val_s, val_r;

        if (id) {
            val_s = res[1];

            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);

            res[0] = val_r;

            res[1] = solve(res, initial, 1, time_interval, 0);

            val_s = res[1];

            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);

            res[1] = 0.1 + (val_s + val_r) / 2.0;

        } else {
            val_s = res[N - 2];

            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            res[N - 1] = val_r;

            res[N - 2] = solve(res, initial, N - 2, time_interval, 0);
            val_s = res[N - 2];

            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            res[N - 2] = 0.1 + (val_s + val_r) / 2.0;

        }

        MPI_Barrier(MPI_COMM_WORLD);

        current_convergence = calc_difference(res, initial, N);

        memcpy(initial, res, N * sizeof(double));

        actual_time += time_interval;
    }
    free(res);
    free(initial);
}



double calc_difference(double *res, double *initial, int N)
{
#ifndef abs
#define abs(val) ((val) < 0 ? -(val) : (val))
#endif
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


double solve(double *actual, double *initial, int index, double time_interval, double alpha)
{
    const double K = 0.01;
    const double int_len = 1.0 / 20.0;
    return (
               K * (
                   alpha * (time_interval / sqr(int_len)) * (actual[index + 1] + actual[index - 1])
                   + (1 - alpha) * (time_interval / sqr(int_len)) *
                   (initial[index + 1] - 2 * initial[index] + initial[index - 1])
               ) + initial[index]
           ) / (1 + (2 * K * alpha * time_interval) / sqr(int_len) );
}

void print_double_vector(double *vector, int N, int start, int stop)
{
    for (int i = start; i < stop; ++i) {
        printf("%f ", vector[i]);
    }
    //printf("\n");
}