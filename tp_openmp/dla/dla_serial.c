#include "dla.h"

int main(int argc, char const* argv[])
{
    if (argc < 2) {
        printf("Cantidad de particulas\n");
        return 1;
    }

    int Points;
    sscanf(argv[1], "%d", &Points);

    int size = 1024;

    char* board = (char*) calloc(size * size, sizeof(char));

    for (int i = 0; i < size; ++i) {
        board[i * size] = 1;
    }

    struct drand48_data buff;
    time_t seed;
    time(&seed);
    srand48_r((int)seed, &buff);

    //    struct timeval begin, end;
    //  gettimeofday(&begin, NULL);

    for (int p = 0; p < Points; ++p) {
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

                    if (((size * size) > (col + row)) && board[col + row]) {
                        found = true;
                        break;
                    }
                }
            }
        }
        int pos = x + size * y;

        board[x + size * y] = 1;

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



