#include <stdio.h>
#include <smmintrin.h> /* header file for the SSE intrinsics we gonna use */
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

/*
int mul_vectorized( int n, float *a )
{
    __m128f mul = _mm_setzero_si128();
    __m128f* ai = (__m128*) a;
    int index = 0;
    int temp_index = 0;
    int mul1[4];
    for(index = 0; index < n/4*4; index += 4){
	ai = (__m128*) a;
	mul = _mm_mul_ps(mul, _mm_loadu_ps(ai));
	a += 4;
    }
    _mm_storeu_ps((__m128*) mul1, mul);
    int mul2 = mul1[0]+mul1[1]+mul1[2]+mul1[3];
    while(index < n){
	mul2 += a[temp_index];
	index += 1;
	temp_index += 1;
    }
    return mul2;
}
*/
/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_sgemm (int n, float* A, float* B, float* C)
{
    int blocksize = 32;
    float *At = (float*)malloc(n*n*sizeof(float) );
    transpose( n, 13, At, A );

    int old_n = n;
    /* For each row i of A */
    //for (int i = 0; i < n; i+=blocksize) {
        /* For each column j of B */
        //for (int j = 0; j < n; j+=blocksize) {
        
            float flres[1];
            for(int j_block = 0; j_block/* < i+blocksize && i_block*/ < n; j_block++) {
                for( int k_block = 0; k_block < (n/32*32); k_block+=32) {
                    __m128 res2 = _mm_loadu_ps((B + (k_block + j_block*n)));
                    __m128 res3 = _mm_loadu_ps((B + 4 + (k_block + j_block*n)));
                    __m128 res4 = _mm_loadu_ps((B + 8 + (k_block + j_block*n)));
                    __m128 res5 = _mm_loadu_ps((B + 12 + (k_block + j_block*n)));
                    __m128 res6 = _mm_loadu_ps((B + 16 + (k_block + j_block*n)));
                    __m128 res7 = _mm_loadu_ps((B + 20 + (k_block + j_block*n)));
                    __m128 res8 = _mm_loadu_ps((B + 24 + (k_block + j_block*n)));
                    __m128 res9 = _mm_loadu_ps((B + 28 + (k_block + j_block*n)));
                    for(int i_block = 0; i_block/* < j+blocksize && j_block*/ < n; i_block++) {
                        /* Compute C(i,j) */
                        __m128 cij = _mm_load_ss(C + (i_block+j_block*n));
                        //for( int k_block = 0; k_block < (n/32*32); k_block+=32) {
                            __m128 res = _mm_mul_ps(res2, _mm_loadu_ps(At + (k_block + i_block * n)));
                            res = _mm_add_ps(res,_mm_mul_ps(res3, _mm_loadu_ps(At + 4 + (k_block + i_block * n))));
                            res = _mm_add_ps(res,_mm_mul_ps(res4, _mm_loadu_ps(At + 8 + (k_block + i_block * n))));
                            res = _mm_add_ps(res,_mm_mul_ps(res5, _mm_loadu_ps(At + 12 +  (k_block + i_block * n))));
                            res = _mm_add_ps(res,_mm_mul_ps(res6, _mm_loadu_ps(At + 16 +  (k_block + i_block * n))));
                            res = _mm_add_ps(res,_mm_mul_ps(res7, _mm_loadu_ps(At + 20 +  (k_block + i_block * n))));
                            res = _mm_add_ps(res, _mm_mul_ps(res8, _mm_loadu_ps(At + 24 +  (k_block + i_block * n))));
                            res = _mm_add_ps(res,_mm_mul_ps(res9, _mm_loadu_ps(At + 28 +  (k_block + i_block * n))));
                            res = _mm_hadd_ps(res,res);
                            res = _mm_hadd_ps(res,res);
                            cij = _mm_add_ss(cij,res);
                        //}
                        _mm_store_ss(C + (i_block+j_block*n), cij);
                    }
                }
            }
        //}
    //}
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
