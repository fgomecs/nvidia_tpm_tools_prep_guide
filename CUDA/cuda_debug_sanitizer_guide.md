# CUDA Debugging and Sanitization Guide

*Date:* 2025-05-12

This guide documents a hands-on session using `cuda-gdb` and `compute-sanitizer` for debugging CUDA applications on a Linux VM with NVIDIA GPU support.

---

## 🧪 Environment Setup

- **OS:** Ubuntu on Crusoe (Brev.cloud)
- **GPU Driver Version:** 535.183.06
- **CUDA Versions Installed:** 12.2, 12.4
- **Current Toolkit Used:** 12.2 (to match driver compatibility)

### ✅ Paths Set

```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
```

---

## 🔧 Debugging with `cuda-gdb`

### 1. Compile with Debug Symbols

```bash
nvcc -g -G -o test_debug test.cu
```

### 2. Launch Debugger

```bash
cuda-gdb ./test_debug
```

### 3. Set Breakpoint and Run

```bash
(cuda-gdb) break helloWorld
(cuda-gdb) run
(cuda-gdb) info cuda kernels
(cuda-gdb) cuda thread 0
(cuda-gdb) backtrace
(cuda-gdb) quit
```

### Notes:

- `helloWorld` is the name of the CUDA kernel.
- Use `break <function>` to set breakpoints.
- You can inspect specific CUDA threads, blocks, and stack traces.

---

## 🛠️ Sanitizing with `compute-sanitizer`

### 1. Create Buggy Code

```cpp
// test_w_bug.cu
__global__ void buggyKernel(int* a) {
    int i = threadIdx.x;
    a[i + 10] = i; // Out-of-bounds write
}

int main() {
    int* d_a;
    cudaMalloc(&d_a, 10 * sizeof(int));
    buggyKernel<<<1, 5>>>(d_a);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    return 0;
}
```

### 2. Compile Without `-G`

```bash
nvcc -lineinfo -o test_w_bug test_w_bug.cu
```

### 3. Run with Compute Sanitizer

```bash
compute-sanitizer ./test_w_bug
```

### Output Summary:

- Multiple invalid global writes detected.
- Errors point to `buggyKernel()` exceeding allocated memory bounds.
- Includes file, line, address, and thread/block info.

---

## 🧠 Key Learnings

- **cuda-gdb** is great for stepping through GPU code and inspecting thread state.
- **compute-sanitizer** is effective for memory and thread error detection.
- Always match your CUDA toolkit version to your driver version.
- Avoid mixing incompatible flags (`-G` vs `--generate-line-info`).

---

## 📦 Helpful Tools

- `cuda-gdb`: Interactive GPU debugger.
- `compute-sanitizer`: Static and runtime memory/thread error detection.
- `nvcc -lineinfo`: Enables line-level error location in sanitizer output.

---

## ✅ Next Steps

- Try shared memory violations.
- Explore warp divergence and thread sync bugs.
- Store logs and mark common pitfalls.

---

*Generated with ❤️ using ChatGPT & Brev VM*
