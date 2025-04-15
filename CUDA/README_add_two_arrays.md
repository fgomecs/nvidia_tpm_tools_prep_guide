Here's a simple CUDA program that adds two arrays on your NVIDIA GTX 1060 GPU. I'll include detailed comments explaining each line of the code.

```cpp
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
```

### Explanation of the Code:

1. **Include Headers**: 
   - `#include <iostream>`: Includes the standard input-output stream library for printing results.

2. **CUDA Kernel Function**:
   - `__global__ void add(int *a, int *b, int *c, int n)`: Defines a CUDA kernel function named `add` that will run on the GPU. It takes pointers to arrays `a`, `b`, and `c`, and an integer `n` for the size of the arrays.
   - `int index = threadIdx.x + blockIdx.x * blockDim.x;`: Calculates the global index of the thread. `threadIdx.x` is the thread index within the block, `blockIdx.x` is the block index, and `blockDim.x` is the number of threads per block.
   - `if (index < n) { c[index] = a[index] + b[index]; }`: Checks if the index is within bounds and performs the addition of corresponding elements from arrays `a` and `b`, storing the result in array `c`.

3. **Main Function**:
   - `const int arraySize = 5;`: Defines the size of the arrays.
   - `const int arrayBytes = arraySize * sizeof(int);`: Calculates the number of bytes needed for the arrays.
   - `int h_a[arraySize] = {1, 2, 3, 4, 5};`: Initializes the host array `h_a`.
   - `int h_b[arraySize] = {10, 20, 30, 40, 50};`: Initializes the host array `h_b`.
   - `int h_c[arraySize];`: Declares the host array `h_c` to store the result.

4. **Device Memory Allocation**:
   - `cudaMalloc((void**)&d_a, arrayBytes);`: Allocates memory on the GPU for array `d_a`.
   - `cudaMalloc((void**)&d_b, arrayBytes);`: Allocates memory on the GPU for array `d_b`.
   - `cudaMalloc((void**)&d_c, arrayBytes);`: Allocates memory on the GPU for array `d_c`.

5. **Data Transfer from Host to Device**:
   - `cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);`: Copies data from host array `h_a` to device array `d_a`.
   - `cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);`: Copies data from host array `h_b` to device array `d_b`.

6. **Kernel Launch**:
   - `add<<<1, arraySize>>>(d_a, d_b, d_c, arraySize);`: Launches the kernel with one block of `arraySize` threads.

7. **Data Transfer from Device to Host**:
   - `cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);`: Copies the result from device array `d_c` to host array `h_c`.

8. **Result Output**:
   - Prints the result stored in `h_c`.

9. **Free Device Memory**:
   - `cudaFree(d_a);`, `cudaFree(d_b);`, `cudaFree(d_c);`: Frees the allocated memory on the GPU.

This code demonstrates a basic CUDA program that performs element-wise addition of two arrays using the GPU.
