#include <iostream>
#include <immintrin.h>
#include <sys/time.h>
#include <string>

using namespace std;

const int n = 3000;
// int **A;
unsigned int A[n][n];
__m128 m1,m2,m3;

void Init(){
	// A = new int*[n];
    // for(int i = 0;i < n; i++)
    //     A[i] = new int[n];
    for(int i = 0;i < n; i++){
        for(int j = 0;j < n; j++)
            A[i][j] = rand();
    }
}
void Serial_LU(){
    for(int k = 0;k < n; k++){
        int j = k + 1;
        for(;j < n; j++){
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            for(j = k + 1;j < n; j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void Part1_SIMD_LU(){
	for(int k = 0;k < n; k++){
        m2 = _mm_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        for(;j + 4 < n; j += 4){
            m1 = _mm_loadu_ps((float*)&(A[k][j]));
            m1 = _mm_div_ps(m1,m2);
            _mm_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
			for(j = k + 1;j < n; j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void Part2_SIMD_LU(){
	for(int k = 0;k < n; k++){
        int j = k + 1;
        for(;j < n; j++){
			A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            m3 = _mm_loadu_ps((float*)&(A[i][k]));
            for(j = k + 1;j + 4 < n;j += 4){
                m2 = _mm_loadu_ps((float*)&(A[k][j]));
                m1 = _mm_loadu_ps((float*)&(A[i][j]));
                m2 = _mm_mul_ps(m2,m3);
                m1 = _mm_sub_ps(m1,m2);
                _mm_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}	

void SIMD_LU(){
    for(int k = 0;k < n; k++){
	    m2 = _mm_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        for(;j + 4 < n; j += 4){
			m1 = _mm_loadu_ps((float*)&(A[k][j]));
	        m1 = _mm_div_ps(m1,m2);
            _mm_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
	        m3 = _mm_loadu_ps((float*)&(A[i][k]));
            for(j = k + 1;j + 4 < n;j += 4){
	        	m2 = _mm_loadu_ps((float*)&(A[k][j]));
	        	m1 = _mm_loadu_ps((float*)&(A[i][j]));
	        	m2 = _mm_mul_ps(m2,m3);
	        	m1 = _mm_sub_ps(m1,m2);
                _mm_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}
void Align_SIMD_LU(){
    for(int k = 0;k < n; k++){
        m2 = _mm_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
	    while((k*n+j) % 4 != 0) {
	        A[k][j] = A[k][j] * 1.0 / A[k][k];
                j++;
	    }
        for(;j + 4 < n; j += 4){
            m1 = _mm_loadu_ps((float*)&(A[k][j]));
            m1 = _mm_div_ps(m1,m2);
            _mm_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            m3 = _mm_loadu_ps((float*)&(A[i][k]));
            int j = k + 1;
    	    while((k * n + j) % 4 != 0) {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }
            for(;j + 4 < n;j += 4){
                m2 = _mm_loadu_ps((float*)&(A[k][j]));
                m1 = _mm_loadu_ps((float*)&(A[i][j]));
                m2 = _mm_mul_ps(m2,m3);
                m1 = _mm_sub_ps(m1,m2);
                _mm_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}
int main()
{
	cout << "size: " << n << endl;
    struct timeval head;
    struct timeval tail;
    Init();
    gettimeofday(&head,NULL);
	Serial_LU();
    gettimeofday(&tail,NULL);
	cout << "Serial Time: " << (tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
    Part1_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout << "Part1_SIMD Time: " << (tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
    Part2_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout << "Part2_SIMD Time: " << (tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
    SIMD_LU();
    gettimeofday(&tail,NULL);
    cout << "Simd Time: " << (tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
	Align_SIMD_LU();
    gettimeofday(&tail,NULL);
	cout << "Align_SIMD Time: " << (tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
    return 0;
}

