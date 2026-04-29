/*
 * Copyright (c) 2022 The Regents of the University of California
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 150

typedef struct {
    int start_row;
    int end_row;
} WorkerArgs;

static int first[MATRIX_SIZE][MATRIX_SIZE];
static int second[MATRIX_SIZE][MATRIX_SIZE];
static int multiply[MATRIX_SIZE][MATRIX_SIZE];

static void *multiply_rows(void *arg)
{
    WorkerArgs *work = (WorkerArgs *)arg;

    for (int c = work->start_row; c < work->end_row; c++) {
        for (int d = 0; d < MATRIX_SIZE; d++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += first[c][k] * second[k][d];
            }
            multiply[c][d] = sum;
        }
    }

    return NULL;
}

static void populate_matrices(void)
{
    for (int x = 0; x < MATRIX_SIZE; x++) {
        for (int y = 0; y < MATRIX_SIZE; y++) {
            first[x][y] = x + y;
            second[x][y] = (4 * x) + (7 * y);
        }
    }
}

static long checksum_matrix(void)
{
    long sum = 0;
    for (int x = 0; x < MATRIX_SIZE; x++) {
        for (int y = 0; y < MATRIX_SIZE; y++) {
            sum += multiply[x][y];
        }
    }

    return sum;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <iterations> <num_cores>\n", argv[0]);
        return 1;
    }

    char *endptr = NULL;
    long iterations = strtol(argv[1], &endptr, 10);
    if (*argv[1] == '\0' || *endptr != '\0' || iterations <= 0) {
        fprintf(stderr, "Invalid iterations: %s\n", argv[1]);
        return 1;
    }

    endptr = NULL;
    long num_threads_long = strtol(argv[2], &endptr, 10);
    if (*argv[2] == '\0' || *endptr != '\0' || num_threads_long <= 0) {
        fprintf(stderr, "Invalid num_cores: %s\n", argv[2]);
        return 1;
    }

    if (num_threads_long > MATRIX_SIZE) {
        num_threads_long = MATRIX_SIZE;
    }

    int num_threads = (int)num_threads_long;

    pthread_t *threads = malloc((size_t)num_threads * sizeof(*threads));
    WorkerArgs *worker_args = malloc((size_t)num_threads * sizeof(*worker_args));
    if (threads == NULL || worker_args == NULL) {
        fprintf(stderr, "Failed to allocate thread resources\n");
        free(threads);
        free(worker_args);
        return 1;
    }

    printf("Populating the first and second matrix...\n");
    populate_matrices();
    printf("Done!\n");

    long final_sum = 0;
    for (long iter = 0; iter < iterations; iter++) {
        for (int t = 0; t < num_threads; t++) {
            worker_args[t].start_row = (t * MATRIX_SIZE) / num_threads;
            worker_args[t].end_row = ((t + 1) * MATRIX_SIZE) / num_threads;

            if (pthread_create(&threads[t], NULL, multiply_rows, &worker_args[t]) != 0) {
                fprintf(stderr, "pthread_create failed for thread %d\n", t);
                free(threads);
                free(worker_args);
                return 1;
            }
        }

        for (int t = 0; t < num_threads; t++) {
            if (pthread_join(threads[t], NULL) != 0) {
                fprintf(stderr, "pthread_join failed for thread %d\n", t);
                free(threads);
                free(worker_args);
                return 1;
            }
        }

        final_sum = checksum_matrix();
    }

    printf("Iterations: %ld\n", iterations);
    printf("Threads used: %d\n", num_threads);
    printf("The sum is %ld\n", final_sum);

    free(threads);
    free(worker_args);
    return 0;
}
