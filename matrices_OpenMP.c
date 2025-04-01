#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Inicializa una matriz con valores aleatorios en [0,1) */
void init_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[i * cols + j] = (double)rand() / (double)RAND_MAX;
}

/* Pone en cero una matriz */
void zero_matrix(double *M, int rows, int cols) {
    memset(M, 0, rows * cols * sizeof(double));
}

/* Imprime una matriz (sólo si es pequeña) */
void print_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.3f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Multiplicación de matrices: C = A * B */
void mult(double *A, double *B, double *C, int m, int k, int n) {
    #pragma omp for schedule(dynamic) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum=0;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}



/* Suma de matrices: C = A + B */
void sum_matrix(double *A, double *B, double *C, int rows, int cols) {
    #pragma omp for schedule(dynamic) collapse(2) 
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
}



int main(int argc, char **argv) {
    int dim, n_threads;
    double t_start, t_end, time_mult1, time_mult2, time_sum;

    dim = atoi(argv[1]);
    n_threads = atoi(argv[2]);

    double *A = (double *)malloc(dim * dim * sizeof(double));
    double *B = (double *)malloc(dim * dim * sizeof(double));
    double *C = (double *)malloc(dim * dim * sizeof(double));
    double *D = (double *)malloc(dim * dim * sizeof(double));
    double *E = (double *)malloc(dim * dim * sizeof(double));

    // Inicializar A y B con valores aleatorios
    init_matrix(A, dim, dim);
    init_matrix(B, dim, dim);
    // Inicializar C, D, E a cero
    zero_matrix(C, dim, dim);
    zero_matrix(D, dim, dim);
    zero_matrix(E, dim, dim);

    #pragma omp parallel num_threads(n_threads) private(t_start,t_end) reduction(max: time_mult1,time_mult2,time_sum)
    {
        //Operación 1: C = A * B
        t_start=omp_get_wtime();
        mult(A,B,C,dim,dim,dim);
        t_end=omp_get_wtime();
        time_mult1= t_end - t_start;

        //Operación 2: D = C * B
        t_start=omp_get_wtime();
        mult(C,B,D,dim,dim,dim);
        t_end=omp_get_wtime();
        time_mult2= t_end - t_start;

        //Operación 3: E = D + C
        t_start=omp_get_wtime();
        sum_matrix(D,C,E,dim,dim);
        t_end=omp_get_wtime();
        time_sum= t_end - t_start;
    }

    printf("Tiempo de ejecucion de C = A * B: %f segundos\n", time_mult1);
    printf("Tiempo de ejecucion de D = C * B: %f segundos\n", time_mult2);
    printf("Tiempo de ejecucion de E = D + C: %f segundos\n", time_sum);
    printf("Tiempo de ejecucion total: %f segundos\n", time_mult1+time_mult2+time_sum);

    if (dim<=10){
    	printf("\nMatrix A:\n");
        print_matrix(A, dim, dim);
        printf("Matrix B:\n");
        print_matrix(B, dim, dim);
        printf("Matrix C = A * B:\n");
        print_matrix(C, dim, dim);
        printf("Matrix D = C * B:\n");
        print_matrix(D, dim, dim);
        printf("Matrix E = D + C:\n");
        print_matrix(E, dim, dim);
    }

}