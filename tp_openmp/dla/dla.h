#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>


void print_board_1(char* board, int size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (board[i * size + j]) {
                printf("%d %d\n", i, j );
            }
        }
    }
}


int rand_step(int v, struct drand48_data* buff, int size)
{
    long int r;
    lrand48_r(buff, &r);
    int val = v + (r % 3) - 1;
    if (val > 0 && val < size - 1) {
        return val;
    }
    else {
        return v;
    }
}