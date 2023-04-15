#include <iostream>
#include <immintrin.h>
#include <sys/time.h>
#include <climits>
using namespace std;

const int n = 3000;
unsigned int A[n][n];
__m512 m1,m2,m3;


void Init(){
    for(int i = 0;i < n; i++){
        for(int j = 0; j < n; j++)
            A[i][j] = rand() % INT_MAX +1;
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
            for(j = k + 1;j < n;j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void Part1_SIMD_LU(){
    for(int k = 0;k < n; k++){
        m2 = _mm512_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        for(;j + 16 < n; j += 16){
            m1 = _mm512_loadu_ps((float*)&(A[k][j]));
            m1 = _mm512_div_ps(m1,m2);
            _mm512_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
			for(j = k + 1;j < n;j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void Part2_SIMD_LU(){
    for(int k = 0;k < n; k++){
        m2 = _mm512_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        for(;j < n; j++){
			A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            m3 = _mm512_loadu_ps((float*)&(A[i][k]));
            for(j = k + 1;j + 16 < n;j += 16){
                m2 = _mm512_loadu_ps((float*)&(A[k][j]));
                m1 = _mm512_loadu_ps((float*)&(A[i][j]));
                m2 = _mm512_mul_ps(m2,m3);
                m1 = _mm512_sub_ps(m1,m2);
                _mm512_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}


void SIMD_LU(){
    for(int k = 0;k < n; k++){
		m2 = _mm512_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        for(;j + 16 < n; j += 16){
	    	m1 = _mm512_loadu_ps((float*)&(A[k][j]));
	    	m1 = _mm512_div_ps(m1,m2);
            _mm512_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
	    	m3 = _mm512_loadu_ps((float*)&(A[i][k]));
            for(j = k + 1;j + 16 < n;j += 16){
	    		m2 = _mm512_loadu_ps((float*)&(A[k][j]));
	    		m1 = _mm512_loadu_ps((float*)&(A[i][j]));
	    		m2 = _mm512_mul_ps(m1,m2);
	    		m1 = _mm512_sub_ps(m1,m3);
            	_mm512_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}

void Align_SIMD_LU(){
    for(int k = 0;k < n; k++){
        m2 = _mm512_loadu_ps((float*)&(A[k][k]));
        int j = k + 1;
        while((k * n + j) % 16 != 0) {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
            j++;
        }
        for(;j + 16 < n; j += 16){
            m1 = _mm512_loadu_ps((float*)&(A[k][j]));
            m1 = _mm512_div_ps(m1,m2);
            _mm512_storeu_ps((float*)&(A[k][j]),m1);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            m3 = _mm512_loadu_ps((float*)&(A[i][k]));
			while((k * n + j) % 16 != 0) {
            	A[i][j] = A[i][j] - A[i][k] * A[k][j];
            	j++;
        	}
            for(j = k + 1;j + 16 < n;j += 16){
                m2 = _mm512_loadu_ps((float*)&(A[k][j]));
                m1 = _mm512_loadu_ps((float*)&(A[i][j]));
                m1 = _mm512_mul_ps(m1,m2);
                m1 = _mm512_sub_ps(m1,m3);
                _mm512_storeu_ps((float*)&(A[i][j]),m1);
            }
            A[i][k] = 0.0;
        }
    }
}

int main()
{
    struct timeval head;
    struct timeval tail;
    Init();
    gettimeofday(&head,NULL);
    Serial_LU();
    gettimeofday(&tail,NULL);
    cout<<"Serial Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
    Part1_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"Part1 SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
	
	gettimeofday(&head,NULL);
    Part2_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"Part2 SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
	
	gettimeofday(&head,NULL);
    SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;

	gettimeofday(&head,NULL);
    Align_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"Align SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
    return 0;
}

