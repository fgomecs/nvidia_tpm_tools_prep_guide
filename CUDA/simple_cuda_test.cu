#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from CUDA! Thread ID: %d\n", threadIdx.x);
}

int main() {
    helloCUDA<<<1, 5>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed successfully!\n");
    }
    return 0;
}