# NVIDIA AI Software & Hardware Stack Overview

This document outlines the full-stack NVIDIA AI platform, from end-user interfaces down to GPU hardware, based on the architecture you’ve diagrammed. CUDA is treated as the central execution engine, with each layer above and below contributing to building, deploying, and operating AI workloads at scale.

---

## ■ End Users / APIs
```
End Users / APIs (Chatbot Apps, AI Agents, LLM UIs)
```
These are user-facing applications that rely on inference capabilities provided through REST or gRPC APIs. These clients do not know or care how models are trained or optimized — they just send prompts or inputs and receive results.

**Examples:**
- Chatbots (support agents, customer service)
- LLM frontends (e.g., UI for GPT-like assistants)
- AI agents that orchestrate multiple APIs

---

## ▲ Application Layer
```
NVIDIA NIMs (LLMs, CV, ASR)
Custom Apps (Inference, RAG Pipelines)
```
This is the highest software abstraction that interfaces directly with end users. It includes pre-packaged model APIs and custom inference pipelines.

- **NIMs (NVIDIA Inference Microservices):** Containerized, REST/gRPC-exposed endpoints for LLMs, CV, ASR, etc. They abstract away model deployment, serving, and optimization.
- **Custom Applications:** Includes RAG pipelines, CV applications, and autonomous systems. Often built using Python, FastAPI, Flask, or similar frameworks.

These applications send inputs to inference engines and receive predictions, embeddings, detections, or transcriptions in return.

---

## ▲ Orchestration Layer
```
Kubernetes (w/ NVIDIA GPU Operator)
Docker (w/ NVIDIA Container Toolkit)
```
This layer manages the lifecycle of AI workloads across multi-node GPU clusters. It enables portability, scheduling, GPU allocation, and scaling.

- **Kubernetes**: Cluster orchestrator that schedules containers and workloads.
- **NVIDIA GPU Operator**: Automates the deployment of NVIDIA drivers, DCGM, monitoring tools, and MIG configurations on Kubernetes clusters.
- **Docker + NVIDIA Container Toolkit**: Ensures containers can access GPUs via runtime hooks (e.g., `--gpus all`).

These tools enable robust, repeatable deployment of AI applications in production environments.

---

## ▲ Inference Layer
```
Triton Inference Server
TensorRT, ONNX Runtime
```
The inference layer receives models and executes them using optimized runtimes.

- **Triton**: Multi-framework inference server that supports batching, ensemble models, and GPU multi-tenancy. It abstracts model execution and simplifies deployment.
- **TensorRT**: A CUDA-based inference optimizer and runtime that compresses and accelerates models (e.g., converts FP32 to FP16/INT8).
- **ONNX Runtime**: Executes ONNX models (originally exported from PyTorch/TF) with GPU acceleration using CUDA or TensorRT providers.

These tools ensure fast and efficient model execution on NVIDIA GPUs.

---

## ▲ Framework Layer
```
PyTorch, TensorFlow, ONNX Models
```
This is where data scientists and researchers define, train, and export models.

- **PyTorch**: Dynamic graph deep learning framework widely used for training and experimentation.
- **TensorFlow**: Static and dynamic computational graph support, with widespread production use.
- **ONNX**: Interchange format for exporting models to inference engines like TensorRT or Triton.

Frameworks interface with CUDA libraries and automatically offload operations to GPU cores.

---

## ▲ CUDA Libraries
```
cuDNN, cuBLAS, cuTENSOR
```
These libraries provide GPU-accelerated math primitives for frameworks.

- **cuDNN**: Specialized in DNN primitives (convolutions, pooling, activations).
- **cuBLAS**: GPU-accelerated BLAS (Basic Linear Algebra Subprograms) operations.
- **cuTENSOR**: For high-performance tensor contractions, reductions, and transformations.

Used internally by PyTorch, TensorFlow, and ONNX Runtime. Key to accelerating training and inference.

---

## ▲ CUDA Layer
```
CUDA Runtime, nvcc, GPU Driver
```
This is the programming and execution layer for GPUs.

- **CUDA Runtime API**: Abstraction layer for memory management, kernels, and stream execution.
- **nvcc**: NVIDIA CUDA Compiler that turns `.cu` code into PTX, then SASS (GPU-specific assembly).
- **GPU Driver**: Kernel-mode driver that translates instructions and manages hardware access.

CUDA provides the interface through which all AI frameworks and libraries ultimately communicate with the hardware.

---

## ▲ Hardware Layer
```
NVIDIA GPU Hardware (H100, GB200)
```
Where actual computation occurs.

- **CUDA Cores**: Handle scalar operations, basic math.
- **Tensor Cores**: Optimized for matrix multiplication (FP16, INT8, BF16).
- **SFUs**: Special Function Units (e.g., trig, exp, log).
- **HBM3e**: High Bandwidth Memory, local to each GPU die.
- **NVLink**: GPU ↔ GPU & GPU ↔ CPU interconnect.
- **NVSwitch**: Enables full mesh communication between multiple GPUs in a node.

Example architecture:
- **B100**: 1 GPU w/ 2 dies
- **B200**: 1 NVLink-native GPU
- **GB200**: Grace CPU + 2x B200s (4 GPU dies) + NVSwitch + NVMe

---

## ■ Storage Layer
```
Persistent Storage (e.g., Portworx, EBS)
Model Catalogs, Data Lakes
```
Used to store datasets, pretrained models, and logs.

- **Local NVMe**: Fast I/O during training or inference.
- **Networked Storage (NFS, EBS, Portworx)**: Shared volumes across nodes.
- **GPUDirect Storage**: Enables direct memory access from storage → GPU bypassing CPU for low-latency data loading.
- **Catalogs**: NGC (NVIDIA GPU Cloud), Hugging Face Hub, custom artifact repositories.

This layer ensures data availability, versioning, and speed.

---
```mermaid
graph TD
    %% Define subgraphs for layers
    subgraph Application Layer
        A[End Users / APIs<br>Chatbots, AI Agents, LLM UIs]
        B[NIMs / Custom Apps<br>LLMs, CV, ASR, RAG Pipelines]
    end
    subgraph Orchestration Layer
        subgraph Kubernetes Orchestration
            C[Kubernetes<br>Scheduler, NVIDIA GPU Operator]
        end
        subgraph Containerization
            D[Docker<br>w/ NVIDIA Container Toolkit]
        end
    end
    subgraph Inference Layer
        E[Triton Inference Server]
        F[TensorRT]
        G[ONNX Runtime]
    end
    subgraph Framework Layer
        H[PyTorch / TensorFlow / ONNX Models]
    end
    subgraph CUDA Libraries
        I[cuDNN / cuBLAS / cuTENSOR]
    end
    subgraph CUDA Layer
        J[CUDA Runtime / GPU Driver]
        K[nvcc Output<br>Grids, Blocks, Threads]
    end
    subgraph Hardware Layer
        L[NVIDIA GPU Hardware]
        M[SM<br>Streaming Multiprocessor]
        N[CUDA Cores<br>Scalar, General-Purpose Compute]
        O[Tensor Cores<br>Matrix Ops for AI/HPC]
        P[SFU<br>Transcendental/Math Functions]
        Q[Warps<br>32 Threads]
        R[SASS<br>Shader Assembly]
        S[PSX<br>Programmable Switch]
    end
    subgraph Storage Layer
        T[Persistent Storage<br>Model Catalogs, Data Lakes, NVMe]
    end

    %% Define interactions with labels
    A -->|REST/gRPC APIs| B
    B -->|Deploys to| D
    D -->|Packages Containers| C
    C -->|Schedules/Orchestrates| E
    E -->|Optimizes with| F
    E -->|Executes with| G
    E -->|Serves Models| H
    H -->|Uses| I
    I -->|Accelerates| J
    J -->|Drives| L
    K -->|Defines Execution| Q
    L -->|Contains| M
    M -->|Executes| Q
    Q -->|Scalar Compute| N
    Q -->|Matrix Compute| O
    Q -->|Math Functions| P
    R -->|Instructs| Q
    S -->|Routes| M
    T -->|Stores Models/Data| C
    T -->|Provides Models/Data| E
    E -->|Loads Models/Data| T

    %% Styling for readability
    classDef layer fill:#e6f3ff,stroke:#005577,stroke-width:2px,color:#000000
    classDef hardware fill:#d9f7d9,stroke:#005577,stroke-width:2px,color:#000000
    classDef orch fill:#e6e6fa,stroke:#005577,stroke-width:2px,color:#000000
    class A,B,E,F,G,H,I,J,K,T layer
    class L,M,N,O,P,Q,R,S hardware
    class C,D orch
    style A fill:#b3e0ff
    style B fill:#b3e0ff
    style C fill:#e6e6fa
    style D fill:#dcdcdc
    style E fill:#fffacd
    style F fill:#fffacd
    style G fill:#fffacd
    style H fill:#f0e68c
    style I fill:#add8e6
    style J fill:#87cefa
    style K fill:#87cefa
    style T fill:#f5f5dc
```
