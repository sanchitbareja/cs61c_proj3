#include <stdio.h>
#include <emmintrin.h> /* header file for the SSE intrinsics we gonna use */
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

void transpose( int n, int blocksize, float *dst, float *src ) {
    int i,j,i_block,j_block;
    /* TO DO: implement blocking (two more loops) */
    for( i = 0; i < n; i+=blocksize )
	for( j = 0; j < n; j+=blocksize )
	    for(i_block = i; i_block< i + blocksize && i_block < n; i_block++)
		for(j_block = j; j_block < j+blocksize && j_block < n; j_block++)
		    dst[j_block+i_block*n] = src[i_block+j_block*n];
    
    //from lab7
    //without cache blocking -> 43.296 milliseconds
    //with cache blocking
    // 	block size 2 -> 25.066
    //	block size 20 -> 18.703
    // 	block size 100 -> 8.744
    // 	block size 200 -> 8.414
    // 	block size 400 -> 8.813
    // 	block size 1000 -> 43.078
}


/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_sgemm (int n, float* A, float* B, float* C)
{
    int blocksize = 10;
    float *At = (float*)malloc( n*n*sizeof(float) );
    transpose( n, blocksize, At, A );

    /* For each row i of A */
    for (int i = 0; i < n; i+=blocksize) {
        /* For each column j of B */
        for (int j = 0; j < n; j+=blocksize) {
            //for(int k = 0; k < n; k+= blocksize){
                for(int i_block = i; i_block < i+blocksize && i_block < n; i_block++) {
                    for(int j_block = j; j_block < j+blocksize && j_block < n; j_block++) {
                        /* Compute C(i,j) */
                        float cij = C[i_block+j_block*n];
                        for( int k_block = 0; k_block < n; k_block++) {
                            cij += At[k_block+i_block*n] * B[k_block+j_block*n];
                        }
                        C[i_block+j_block*n] = cij;
                    }
                }
            //}
        }
    }
    free( At );
}

void square_sgemm_naive (int n, float* A, float* B, float* C)
{
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      float cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
	cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
}

/*
int main( int argc, char **argv ) {
    int n = 64,i,j;

    // allocate an n*n block of integers for the matrices
    float *A = (float*)malloc( n*n*sizeof(float) );
    float *B = (float*)malloc( n*n*sizeof(float) );
    float *C = (float*)malloc( n*n*sizeof(float) );
    float *D = (float*)malloc( n*n*sizeof(float) );
    float *E = (float*)malloc( n*n*sizeof(float) );
    float *F = (float*)malloc( n*n*sizeof(float) );

    // initialize A,B to random integers 
    srand48( time( NULL ) );
    for( i = 0; i < n*n; i++ ) A[i] = lrand48( );
    for( i = 0; i < n*n; i++ ) B[i] = lrand48( );
    for( i = 0; i < n*n; i++ ) D[i] = A[i];
    for( i = 0; i < n*n; i++ ) E[i] = B[i];
    for( i = 0; i < n*n; i++ ) C[i] = 0;
    for( i = 0; i < n*n; i++ ) F[i] = 0;
    
    // measure performance 
    struct timeval start, end;

    gettimeofday( &start, NULL );
    square_sgemm( n, A, B, C );
    gettimeofday( &end, NULL );

    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    printf( "%g milliseconds\n", seconds*1e3 );

    gettimeofday( &start, NULL );
    square_sgemm_naive( n, D, E,F );
    gettimeofday( &end, NULL );

    seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    printf( "%g milliseconds\n", seconds*1e3 );

    // check correctness 
    for( i = 0; i < n; i++ )
        for( j = 0; j < n; j++ )
            if( C[j+i*n] != F[j+i*n] ) {
	        printf("Error!!!! MMM does not result in correct answer!! i = %d j = %d C = %f F = %f \n", i, j, C[j+i*n], F[j+i*n]);
	        exit( -1 );
            }
  
    // release resources
    free( A );
    free( B );
    free( C );
    free( D );
    free( E );
    free( F );
    return 0;
}
*/
