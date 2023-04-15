#include <iostream>
#include<arm_neon.h>
#include<sys/time.h>
using namespace std;

const int n = 3000;
float32_t A[n][n];
float32x4_t vakk = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);

void Init(){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            A[i][j] = rand();
    }
}

void Serial_LU() {
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
        float32x4_t vakk = vld1q_dup_f32(&A[k][k]);
        int j = k + 1;
        for(;j + 4 < n; j += 4){
            vakj = vld1q_f32(&A[k][j]);
            vakj = vdivq_f32(vakj,vakk);
            vst1q_f32(&A[k][j],vakj);
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
        float32x4_t vakk = vld1q_dup_f32(&A[k][k]);
        int j = k + 1;
        for(;j < n; j++){
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            for(;j + 4 < n;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vaik = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vaik);
                vst1q_f32(&A[i][j],vaij);
            }
            A[i][k] = 0.0;
        }
    }
}

void SIMD_LU(){
    for(int k = 0;k < n; k++){
        float32x4_t vakk = vld1q_dup_f32(&A[k][k]);
        int j = k + 1;
        for(;j + 4 < n; j += 4){
            vakj = vld1q_f32(&A[k][j]);
            vakj = vdivq_f32(vakj,vakk);
            vst1q_f32(&A[k][j],vakj);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            for(;j + 4 < n;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vaik = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vaik);
                vst1q_f32(&A[i][j],vaij);
            }
            A[i][k] = 0.0;
        }
    }
}
void Align_SIMD_LU(){
    for(int k = 0;k < n; k++){
        float32x4_t vakk = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        while((k * n + j) % 4 != 0){
            A[k][j] = A[k][j] * 1.0 / A[k][k];
            j++;
        }
        for(;j + 4 < n; j += 4){
            vakj = vld1q_f32(&A[k][j]);
            vakj = vdivq_f32(vakj,vakk);
            vst1q_f32(&A[k][j],vakj);
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            j = k + 1;
            while((k * n + j) % 4 != 0){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }
            for(;j + 4 < n;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vaik = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vaik);
                vst1q_f32(&A[i][j],vaij);
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
    cout<<"Part1 SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;
    
    gettimeofday(&head,NULL);
    Part2_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"Part2 SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;
    
    gettimeofday(&head,NULL);
    SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;

    gettimeofday(&head,NULL);
    Align_SIMD_LU();
    gettimeofday(&tail,NULL);
    cout<<"Align SIMD Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;


    return 0;
}
