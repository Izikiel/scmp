#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"

#ifndef sqr
#define sqr(x) (x*x)
#endif

double solve(double* actual, double* initial, int index, double delta_t, double alpha, double delta_x);
void explicit_method_mpi( int N, double time_interval, double left, double right, double total_time);
void explicit_method( int N, double time_interval, double total_time);
void print_double_vector(double* vector, int start, int N, FILE* f);

int main(int argc, char const* argv[])
{
    if (argc < 4) {
        printf("Falta modo si serial o paralelo (S o P), paso tiempo, tiempo total\n");
        return -1;
    }

    char mode = argv[1][0];

    double time_interval;
    sscanf(argv[2], "%lf", &time_interval);

    double total_time;
    sscanf(argv[3], "%lf", &total_time);

    if (mode == 'P') {
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

        double left = id ? 0.1 : 1;
        double right = id ? 0 : 0.1;

        explicit_method_mpi(N, time_interval, left, right, total_time);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Finalize();
    }
    else {
        int N = 20;
        explicit_method(N, time_interval, total_time);
    }


    return 0;
}

void explicit_method( int N, double delta_t, double total_time)
{
    //setting boundary conditions
    double* initial = calloc(N, sizeof(double));
    double* res = calloc(N, sizeof(double));

    initial[0] = res[0] = 1;
    initial[N - 1] = res[N - 1] = 0;

    int half = N / 2 - 1;

    initial[half] = res[half] = 0.1;

    char file_name[20];
    sprintf(file_name, "serial_out.txt");
    FILE* out_data = fopen(file_name, "w");

    for (double actual_time = 0.0; actual_time < total_time; actual_time += delta_t) {
        print_double_vector(res, 0, N, out_data);

        double delta_x = 0.4; // 20.0 / ((double) N);

        for (int i = 1; i < N - 1; ++i) {
            res[i] = solve(res, initial, i, delta_t, 0, delta_x);
        }

        res[half] = 0.1 + (res[half - 1] + res[half + 1]) / 2.0;

        memcpy(initial, res, N * sizeof(double));
    }
    free(res);
    free(initial);
    fclose(out_data);
}


void explicit_method_mpi( int N, double delta_t, double left, double right, double total_time)
{
    //setting boundary conditions
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int dest = id ? 0 : 1;
    int tag = 0;
    MPI_Status status;

    char file_name[20];
    sprintf(file_name, "mpi_out_%d.txt", id);

    FILE* out_data = fopen(file_name, "w");

    int start = 1;
    int stop = N - 1;

    int middle = id ? 0 : N - 1;

    double* initial = calloc(N, sizeof(double));
    double* res = calloc(N, sizeof(double));

    initial[0] = res[0] = left;
    initial[N - 1] = res[N - 1] = right;

    for (double actual_time = 0; actual_time < total_time; actual_time += delta_t) {
        print_double_vector(res, id, N, out_data);

        double delta_x = 0.4; // 20.0 / ((double) N);

        for (int i = start; i < stop; ++i) {
            res[i] = solve(res, initial, i, delta_t, 0, delta_x);
        }

        double val_s = id ? res[1] : res[N - 2];
        double val_r;

        MPI_Barrier(MPI_COMM_WORLD);

        if (id) {
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
        }
        else {
            MPI_Recv(&val_r, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&val_s, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }

        res[middle] = 0.1 + (val_s + val_r) / 2.0;

        memcpy(initial, res, N * sizeof(double));
    }

    free(res);
    free(initial);
    fclose(out_data);
}

inline double solve(double* actual, double* initial, int index, double delta_t, double alpha, double delta_x)
{
    const double K = 0.1;
    const double r = K * (delta_t / sqr(delta_x));
    return (
               r * (
                   alpha *  (actual[index + 1] + actual[index - 1])
                   + (1 - alpha) * (initial[index + 1] - 2 * initial[index] + initial[index - 1])
               ) + initial[index]
           ) / (1 + (2 * K * alpha * delta_t) / sqr(delta_x) );
}

void print_double_vector(double* vector, int start, int N, FILE* f)
{
    for (int i = start; i < N; ++i) {
        fprintf(f, "%f ", vector[i]);
    }
    fprintf(f, "\n");
}