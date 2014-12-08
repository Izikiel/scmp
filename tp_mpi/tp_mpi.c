#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"


#ifndef sqr
#define sqr(x) (x*x)
#endif

double solve(double *actual, double *initial, int index, double time_interval, double alpha);
void explicit_method( int N, double time_interval, double left, double right, double total_time);
// double calc_difference(double *res, double *initial, int N);
void print_double_vector(double *vector, int N, int start, int stop, FILE *f);

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
        explicit_method(N, time_interval, 1, 0, 100);
    } else if (id == 1) {
        explicit_method(N, time_interval, 0, 0, 100);
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

    char file_name[20];
    sprintf(file_name, "mpi_out_%d.txt", id);

    FILE *out_data = fopen(file_name, "w");

    int start = id ? 2 : 1;
    int stop = N - (id ? 1 : 2);

    int middle = id ? 1 : N - 2;
    int edge = id ? 0 : N - 1;


    double *initial = calloc(N, sizeof(double));
    double *res = calloc(N, sizeof(double));

    initial[id] = res[id] = left;
    initial[N - 2 + id] = res[N - 2 + id] = right;

    double actual_time = 0.0;

    while (actual_time < total_time) {

        print_double_vector(res, N, id , stop + 1, out_data);

        for (int i = start; i < stop; ++i) {
            res[i] = solve(res, initial, i, time_interval, 0);
        }

        double val_s = id ? res[1] : res[N - 2];
        double val_r;

        MPI_Barrier(MPI_COMM_WORLD);

        if (id) {
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }

        res[middle + (id ? -1 : 1)] = val_r;

        res[middle] = solve(res, initial, middle, time_interval, 0);

        val_s = res[middle];

        if (id) {
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }

        if ((val_r + val_s) > 0.0) {
            res[middle] = 0.1 + (val_s + val_r) / 2.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        memcpy(initial, res, N * sizeof(double));

        actual_time += time_interval;
    }

    free(res);
    free(initial);
    fclose(out_data);
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

void print_double_vector(double *vector, int N, int start, int stop, FILE *f)
{
    for (int i = start; i < stop; ++i) {
        fprintf(f, "%f ", vector[i]);
    }
    fprintf(f, "\n");
}