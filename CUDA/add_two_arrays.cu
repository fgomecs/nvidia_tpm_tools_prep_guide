// Add two arrays
#include <iostream>
// CUDA kernel function to add elements of two arrays
__global__ void add(int *a, int *b, int *c, int n) {
    // Calculate the global index of the thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Ensure the index is within the bounds of the arrays
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int arraySize = 5;
    const int arrayBytes = arraySize * sizeof(int);

    // Host arrays
    int h_a[arraySize] = {1, 2, 3, 4, 5};
    int h_b[arraySize] = {10, 20, 30, 40, 50};
    int h_c[arraySize];

    // Device arrays
    int *d_a, *d_b, *d_c;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, arrayBytes);
    cudaMalloc((void**)&d_b, arrayBytes);
    cudaMalloc((void**)&d_c, arrayBytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);

    // Launch the kernel with one block of arraySize threads
    add<<<1, arraySize>>>(d_a, d_b, d_c, arraySize);

    // Copy the result from device to host
    cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}