/*
  Partners
  Name1: Eric Atkinson
  Login1: cs61c-er

  Name2: Sanchit Bareja
  Login2: cs61c-ka
  
 */
#include <stdio.h>
#include <smmintrin.h> /* header file for the SSE intrinsics we gonna use */
#include <stdlib.h>
#include <time.h>
#include <string.h>
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
    int new_n = n + 16 - ((n % 16 == 0) ? 16 : n%16);
    static float new_A[1024 * 1024] __attribute__((aligned(16)));
    static float new_B[1024 * 1024] __attribute__((aligned(16)));
    static float new_C[1024 * 1024];__attribute__((aligned(16)));

    for(int i = 0; i < n; i++)
    {
       memcpy(new_A+i*new_n,A+i*n,n*sizeof(float)); 
       memcpy(new_B+i*new_n,B+i*n,n*sizeof(float)); 
       memcpy(new_C+i*new_n,C+i*n,n*sizeof(float)); 

       memset(new_A+n+i*new_n,0,(new_n-n)*sizeof(float));
       memset(new_B+n+i*new_n,0,(new_n-n)*sizeof(float));
       memset(new_C+n+i*new_n,0,(new_n-n)*sizeof(float));
    }


    for(int i = n; i < new_n; i++)
    {
        memset(new_A + i*new_n, 0, new_n * sizeof(float));
        memset(new_B + i*new_n, 0, new_n * sizeof(float));
        memset(new_C + i*new_n, 0, new_n * sizeof(float));
    }

    float* old_A = A;
    A = new_A;
    float* old_B = B;
    B = new_B;
    float* old_C = C;
    C = new_C;
    int old_n = n;
    n = new_n;

    for(int j_block = 0; j_block < n; j_block++) {

        for(int i_block = 0; i_block < n/16*16; i_block+=16) {
            __m128 cij = _mm_loadu_ps(C + i_block + j_block * n);
            __m128 cij2 = _mm_loadu_ps(C + i_block + 4 + j_block * n);
            __m128 cij3 = _mm_loadu_ps(C + i_block + 8 + j_block * n);
            __m128 cij4 = _mm_loadu_ps(C + i_block + 12 + j_block * n);
            for(int k_block = 0; k_block < n/4*4; k_block+= 4){
                cij = _mm_add_ps(cij, _mm_mul_ps(_mm_load1_ps(B + k_block+ j_block * n),_mm_loadu_ps(A + i_block + (k_block) * n)));
                cij = _mm_add_ps(cij, _mm_mul_ps(_mm_load1_ps(B +1+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + (k_block+1) * n)));
                cij = _mm_add_ps(cij, _mm_mul_ps(_mm_load1_ps(B +2+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + (k_block+2) * n)));
                cij = _mm_add_ps(cij, _mm_mul_ps(_mm_load1_ps(B +3+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + (k_block+3) * n)));

                cij2 = _mm_add_ps(cij2, _mm_mul_ps(_mm_load1_ps(B + k_block+ j_block * n),_mm_loadu_ps(A + i_block + 4 + (k_block) * n)));
                cij2 = _mm_add_ps(cij2, _mm_mul_ps(_mm_load1_ps(B +1+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 4 + (k_block+1) * n)));
                cij2 = _mm_add_ps(cij2, _mm_mul_ps(_mm_load1_ps(B +2+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 4+ (k_block+2) * n)));
                cij2 = _mm_add_ps(cij2, _mm_mul_ps(_mm_load1_ps(B +3+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 4+ (k_block+3) * n)));
                
                cij3 = _mm_add_ps(cij3, _mm_mul_ps(_mm_load1_ps(B + k_block+ j_block * n),_mm_loadu_ps(A + i_block + 8 + (k_block) * n)));
                cij3 = _mm_add_ps(cij3, _mm_mul_ps(_mm_load1_ps(B +1+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 8 + (k_block+1) * n)));
                cij3 = _mm_add_ps(cij3, _mm_mul_ps(_mm_load1_ps(B +2+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 8+ (k_block+2) * n)));
                cij3 = _mm_add_ps(cij3, _mm_mul_ps(_mm_load1_ps(B +3+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 8+ (k_block+3) * n)));

                cij4 = _mm_add_ps(cij4, _mm_mul_ps(_mm_load1_ps(B + k_block+ j_block * n),_mm_loadu_ps(A + i_block + 12 + (k_block) * n)));
                cij4 = _mm_add_ps(cij4, _mm_mul_ps(_mm_load1_ps(B +1+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 12 + (k_block+1) * n)));
                cij4 = _mm_add_ps(cij4, _mm_mul_ps(_mm_load1_ps(B +2+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 12 + (k_block+2) * n)));
                cij4 = _mm_add_ps(cij4, _mm_mul_ps(_mm_load1_ps(B +3+ k_block+ j_block * n),_mm_loadu_ps(A + i_block + 12 + (k_block+3) * n)));


            }
            _mm_store_ps((C + i_block +j_block * n), cij);
            _mm_store_ps((C + i_block + 4 + j_block * n),cij2);
            _mm_store_ps((C + i_block + 8 + j_block * n),cij3);
            _mm_store_ps((C + i_block + 12 + j_block * n),cij4);
        }
    }

    for(int i = 0; i < old_n; i++){
        memcpy (old_C + i * old_n, C + i*n, old_n * sizeof(float));
        memcpy (old_B + i * old_n, B + i*n, old_n * sizeof(float));
        memcpy (old_A + i * old_n, A + i*n, old_n * sizeof(float));
    }

    //}
    //}
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
