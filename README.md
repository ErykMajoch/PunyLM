# PunyLM

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Platform: Windows | macOS | Linux](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

A decoder-only language model written from scratch in modern C++20. Features a reverse-mode autograd engine over a
multi-dimensional `Tensor` type, a CPU reference backend (Eigen) and an optional CUDA backend
for GPU acceleration. Designed for education, clarity and composability.

## Features

- **Tape-based autograd**: Every `Tensor` records its computation graph. `backward()` topologically sorts and walks it in reverse, accumulating gradients through closures stored per node.
- **Dual backend architecture**: Operations dispatch on the device of their inputs (`Device::CPU` / `Device::CUDA`). The CPU path is always compiled whilst the CUDA path is an optional plug-in enabled through `-DPUNYLM_ENABLE_CUDA=ON`.
- **Llama-style decoder-only transformer**: RMSNorm, rotary positional embeddings (RoPE), SwiGLU activation in the MLP, pre-norm residual connections and weight-tied input/output embeddings.
- **Two tokenisers**: A simple character-level tokeniser for debugging and small-scale tests and a trainable byte-level BPE tokeniser (GPT-2-style pre-tokenisation) for general usage.
- **AdamW optimiser**: With linear warmup, cosine decay schedule and global-norm gradient clipping.
- **Text generation**: Greedy, temperature, top-k and top-p sampling backed by a cached KV cache.
- **Numerical gradient checking**: Every primitive operation is verified against finite differences to ~1e-3 relative error before the model is assembled.
- **Cross-platform CMake build**: Dependencies pulled with `FetchContent`. Builds on Windows, macOS and Linux.

## Architecture

**Autograd engine.** A `Tensor` is a shared handle to a `TensorImpl` that owns a device-tagged `Storage` (float32 buffer) and an optional `backward_fn` closure. Each primitive operation constructs a new output tensor, links it to its parents and installs a closure that pushes gradients backward using the device's kernel implementation. Calling `backward()` on the loss runs a post-order DFS to produce a topological ordering, seeds the root gradient with ones, then walks the graph in reverse executing each closure. Gradients accumulate (`+=`) so tensors used in multiple positions receive the correct combined contribution.

**Backend dispatch.** The `Device` enum (`CPU` or `CUDA`) tags every tensor's storage. Operations in `punylm::ops` use a `PUNYLM_DISP` macro that routes kernel calls to `punylm::backend::cpu::*` or, when enabled, `punylm::backend::cuda::*` (defined in `.cu` files compiled by `nvcc`). The autograd graph, model, tokeniser, optimiser and training loop are entirely device-agnostic. A single `model.to(Device::CUDA)` moves all parameters to the GPU.

**Model.** The architecture follows the modern Llama-style aapproach. RMSNorm (no learned bias or mean subtraction), rotary positional embeddings applied directly to query and key tensors, SwiGLU gated activation in the feed-forward network, pre-norm residual connections around both attention and MLP sub-layers and weight-tied embedding tables (the same tensor serves as both the input embedding and the output projection head).

**One sequence at a time.** The forward pass processes a single token sequence (no batch dimension in the math). Effective batching is achieved through gradient accumulation in the training loop. `accum` sequences are run forward/backward, gradients accumulate across them, then the optimiser takes a single step. This keeps the attention implementation simple (no permute-free batching) and is documented as an explicit later optimisation.

## Reference Configurations

| Config     | d_model | Layers | Heads | Context | Vocab | Params | Runs On | Time to Train |
|------------|--------:|-------:|------:|--------:|------:|-------:|---------|---------------|
| `smoke`    |     128 |      4 |     4 |     128 |   512 | ~1.6 M | CPU     | Minutes       |
| `coherent` |     512 |      8 |     8 |     512 |  8192 | ~29 M  | GPU     | Hours         |

The `smoke` configuration is used for CI, unit tests and rapid iteration. The `coherent`
configuration *should* (need to check when implemented 😅) fit in 8 GB of VRAM (FP32 weights + AdamW state + activations) and
produce readable TinyStories-style prose after training.

## Quick Start

> Quick start will be provided when the implementation is ready!

## Building

### Prerequisites

- **CMake** >= 3.25
- A **C++20 compiler**: MSVC 2022, Clang >= 14 or GCC >= 12
- (Optional) **CUDA Toolkit** >= 12.x for the CUDA backend

### CPU Build (all platforms)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CUDA Build (Windows / Linux, CUDA Toolkit required)

```bash
cmake -S . -B build-cuda -DPUNYLM_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

### CMake Options

| Option                    | Default | Description                                  |
|---------------------------|--------:|----------------------------------------------|
| `PUNYLM_ENABLE_CUDA`      |    OFF  | Build the optional CUDA backend              |
| `PUNYLM_BUILD_TESTS`      |    OFF  | Build the Catch2 test suite                  |
| `PUNYLM_ENABLE_ASAN`      |    OFF  | Enable AddressSanitizer / UBSan (Linux/macOS)|
| `PUNYLM_ENABLE_COVERAGE`  |    OFF  | Enable code coverage instrumentation         |
| `PUNYLM_WERROR`           |    OFF  | Treat compiler warnings as errors            |

## Current Status

PunyLM is a work in progress project! See the roadmap for currently implemented and future features.

### Roadmap

- [x] Build system, tooling, dependencies
- [ ] Core: device abstraction, storage, RNG, autograd `Tensor` with backward tape
- [ ] Backend operations: CPU kernels for the primitive autograd operations
- [ ] NN modules: RoPE, multi-head causal attention, SwiGLU MLP, transformer block, `PunyLM` model
- [ ] Tokeniser: character level and byte-level BPE (train / encode / decode / persist)
- [ ] Data pipeline: corpus tokenisation to `.bin`, dataset batcher, download script
- [ ] Optimiser: AdamW with linear warmup, cosine decay and global-norm gradient clipping
- [ ] Training loop: gradient accumulation, checkpoint save/resume, CSV logging
- [ ] Generation CLI: greedy, temperature, top-k and top-p sampling with KV cache
- [ ] Tests: numerical gradient checking for every operation, tokeniser round-trips, serialisation round-trips, overfit smoke test
- [ ] CUDA backend: elementwise kernels, cuBLAS matmul, fused softmax, RMSNorm, embedding, optimiser step
- [ ] CI (GitHub Actions): 3 OS build+test matrix, linting, sanitisers, coverage, training smoke test


## Author

Eryk Majoch

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
