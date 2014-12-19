#include "dla.h"

typedef struct {
    char* board;
    int size;
} particle_data;

void* particle_move(void* data);

int main(int argc, char const* argv[])
{
    if (argc < 2) {
        printf("Cantidad de particulas\n");
        return 1;
    }

    int Points;
    sscanf(argv[1], "%d", &Points);

    int size = 1024;

    pthread_t particles[Points];

    char* board = (char*) calloc(size * size, sizeof(char));

    particle_data d;

    d.board = board;
    d.size = size;

    for (int i = 0; i < size; ++i) {
        board[i * size] = 1;
    }

    // struct timeval begin, end;
    // gettimeofday(&begin, NULL);

    for (int p = 0; p < Points; ++p) {
        if (pthread_create(&particles[p], NULL, particle_move, (void*) &d)) {
            printf("Error con el thread Nº%d\n", p);
            return 1;
        }
    }

    for (int i = 0; i < Points; ++i) {
        pthread_join(particles[i], NULL);
    }

    // gettimeofday(&end, NULL);

    // int diff = end.tv_usec - begin.tv_usec;
    // if (diff >= 0) {
    //     printf("%d\n", diff);
    // }

    print_board_1(board, size);

    free(board);
    return 0;
}


void* particle_move(void* data)
{
    particle_data* d = (particle_data*) data;
    int size = d->size;
    struct drand48_data buff;
    time_t seed;
    time(&seed);
    srand48_r((int)seed, &buff);
    double theta;
    drand48_r(&buff, &theta);
    int x = size / 2 + 0.5 * (size - 2) * cos(theta);
    int y = size / 2 + 0.5 * (size - 2) * sin(theta);


    for (bool found = false; !found ;) {

        x = rand_step(x, &buff, size);

        y = rand_step(y, &buff, size);

        for (int i = -1; i <= 1 && !found; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int col = x + i;
                int row = (y + j) * size;

                if (((size * size) > (col + row)) && d->board[col + row]) {
                    found = true;
                    break;
                }
            }
        }
    }
    int pos = x + size * y;

    if (pos < (size * size)) {
        d->board[x + size * y] = 1;
    }
    return NULL;
}
