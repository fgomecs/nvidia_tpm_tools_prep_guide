
# NVIDIA Developer Tools – Full-Stack Debugging & Profiling Flow

## 🔧 Core Tools Overview

| Tool               | Purpose                               | When to Use                      |
|--------------------|----------------------------------------|----------------------------------|
| `cuda-gdb`         | GPU debugger                          | During execution (per-thread/kernel step debugging) |
| `compute-sanitizer` | Detect memory access violations       | During execution (e.g., invalid writes, race conditions) |
| Nsight Systems     | System-wide performance profiling     | During & after execution (trace CPU-GPU interaction, bottlenecks) |
| Nsight Compute     | Kernel-level analysis                 | During execution (instruction stats, memory throughput) |

---

## 📦 Compilation-to-Execution Flow

### Compilation
1. CUDA source (`.cu`) compiled with `nvcc`
2. Developer defines grids, blocks, threads
3. `nvcc` generates PTX, SASS (assembly), and machine code

> ⚙️ Tool Used: `nvcc`

---

### Execution

#### a. Warp Scheduling
- Warp scheduler assigns 32 threads (warp) to CUDA/Tensor cores
- Based on available SPC (Space, Power, Cooling)

#### b. Memory Hierarchy
- Shared memory → L1/L2 cache → HBM → PCIe → CPU RAM
- Latency and throughput are influenced by memory hierarchy decisions

> 🔍 Tools Active:
> - `cuda-gdb`: Set breakpoints, examine variables
> - `compute-sanitizer`: Catch invalid accesses
> - Nsight Systems: Trace memory API latency, CPU↔GPU sync
> - Nsight Compute: Examine instruction-level memory usage

---

## 📊 Tool Integration

### 🔗 CUPTI
- Low-level instrumentation interface
- Collects metrics, events, and traces from CUDA runtime/driver
- Used by Nsight Systems & Compute for internal data gathering

### 🧭 NVTX (NVIDIA Tools Extension)
- Lets developers manually annotate timelines
- Adds human-readable regions to visual traces (e.g., `NVTX_RANGE_PUSH("MatrixMul")`)

### Tools Consuming CUPTI + NVTX:
- Nsight Systems
- Nsight Compute

---

## 🧠 TPM Role Insights for Dev Tools Teams

### 🔍 Typical Developer Team Struggles:
- Debugging obscure kernel crashes
- Detecting race conditions or invalid memory writes
- Latency issues between CPU ↔ GPU
- Improper kernel launch configurations (grid/block sizing)
- Inefficient use of GPU memory hierarchy

### 👷‍♂️ TPM Responsibilities:
- Unblock devs by coordinating debug sessions across tools
- Prioritize profiling/analysis sessions
- Ensure the dev team has reproducible bug reports from sanitizer tools
- Track regressions in performance using Nsight and CUPTI metrics
- Facilitate feedback loop between devs, QA, perf analysis, and product

---

## 🧩 Visual Integration

| Environment | Supported Tools                   |
|-------------|------------------------------------|
| Visual Studio | cuda-gdb, Nsight Systems (plugin) |
| VS Code      | cuda-gdb via terminal, Nsight CLI |
| CLI / SSH    | All tools work with CLI pipelines |

---

## ✅ Best Practices

- Use `nvcc -G -lineinfo` for full debug symbols
- Run `compute-sanitizer` early in dev for correctness
- Switch to `Nsight Systems` for system bottlenecks
- Drill down with `Nsight Compute` for kernel inefficiencies
- Add `NVTX` markers to improve trace clarity

---

## 📂 Version Compatibility

- Ensure `nvcc`, `cuda-gdb`, and `compute-sanitizer` match the installed **CUDA Toolkit** (e.g., 12.2, 12.4)
- Avoid mismatch with driver (check `nvidia-smi`)
- Prefer Nsight tools downloaded separately for latest features

---

## 🔚 Summary Diagram Placement

- **Compilation**: `nvcc` transforms code into PTX/SASS (before warp scheduler)
- **Execution**: Begins at warp scheduling; tools like `cuda-gdb`, sanitizer, and Nsight are used here
- **Post-execution**: Nsight Systems visualizes time traces; Nsight Compute offers kernel-level metrics

---

© 2025 Francisco Gomez – NVIDIA Dev Tools TPM Prep
