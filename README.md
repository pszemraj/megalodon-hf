
# megalodon-hf

This repo contains modeling code representing a pure transformers/torch take on megalodon for learning and simplicity purposes.
- original codebase https://github.com/XuezheMax/megalodon/ (no weights released ever)
- details on the model arch are below, from documentation generated directly from the repo

## Architecture Overview | XuezheMax/megalodon | DeepWiki

URL Source: http://deepwiki.com/XuezheMax/megalodon/1.2-architecture-overview

Markdown Content:
Relevant source files
*   [README.md](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md)
*   [megalodon/csrc/megalodon_extension.cc](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon/csrc/megalodon_extension.cc)

Purpose and Scope
-----------------

This document explains the three-tier architecture of the Megalodon repository, detailing how the Python interface layer, C++ operation layer, and CUDA kernel layer work together to provide high-performance custom operations for the Megalodon language model. This page covers the structural organization, component interactions, and runtime dispatch mechanisms that enable seamless integration between Python and GPU-accelerated code.

For information about specific operations implemented in this architecture, see [Custom CUDA Operations](https://deepwiki.com/XuezheMax/megalodon/3-custom-cuda-operations). For details on the Python APIs and configuration system, see [Python Interface](https://deepwiki.com/XuezheMax/megalodon/2-python-interface). For utility libraries used across all layers, see [Utility Libraries](https://deepwiki.com/XuezheMax/megalodon/4-utility-libraries).

* * *

Three-Tier Architecture
-----------------------

The Megalodon codebase is organized into three distinct architectural layers, each with specific responsibilities:

**Architecture Overview: Three-Tier System Structure**

Sources: [megalodon_extension.cc 1-27](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L1-L27)[README.md 80-152](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md#L80-L152)

* * *

Layer 1: Python Interface
-------------------------

The Python interface layer provides user-facing APIs for model configuration, training, and evaluation. This layer is entirely implemented in Python and interacts with the C++ layer through the `megalodon_extension` module.

### Key Components

| Component | Location | Purpose |
| --- | --- | --- |
| `eval.py` | Root directory | Evaluation script for perplexity and text generation |
| Training code | Referenced in README.md | Training loop with distributed support |
| Configuration classes | `megalodon/config.py` | Dataclasses for model, optimizer, and tokenizer config |
| Package initialization | `megalodon/__init__.py` | Module exports and setup |

The Python layer communicates with custom operations by importing the compiled `megalodon_extension` module and calling its exposed functions. All tensor operations are automatically dispatched to the appropriate implementation (CPU or CUDA) based on tensor device type.

Sources: [README.md 64-78](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md#L64-L78)[README.md 80-152](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md#L80-L152)

* * *

Layer 2: C++ Operation Layer
----------------------------

The C++ operation layer serves as the bridge between Python and high-performance kernel implementations. This layer is implemented using PyBind11 and provides a consistent interface for all custom operations.

### PyBind11 Module Registration

The central integration point is `megalodon_extension.cc`, which registers all operations:

**PyBind11 Operation Registration Flow**

Each operation category is defined in its own header/implementation file pair within the `megalodon/csrc/ops/` directory. The `Define*Op` functions register forward and backward pass functions with the PyBind11 module.

Sources: [megalodon_extension.cc 1-27](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L1-L27)

### Operation Categories

The system implements seven distinct operation categories:

| Operation | Header/Implementation | Forward Function | Backward Function |
| --- | --- | --- | --- |
| Attention | `ops/attention.h/cc` | `attention_fwd` | `attention_bwd` |
| Attention Softmax | `ops/attention_softmax.h/cc` | `attention_softmax_fwd` | `attention_softmax_bwd` |
| EMA Hidden | `ops/ema_hidden.h/cc` | `ema_hidden_fwd` | `ema_hidden_bwd` |
| EMA Parameters | `ops/ema_parameters.h/cc` | `ema_parameters_fwd` | `ema_parameters_bwd` |
| FFT Convolution | `ops/fftconv.h/cc` | `fftconv_fwd` | `fftconv_bwd` |
| Sequence Norm | `ops/sequence_norm.h/cc` | `sequence_norm_fwd` | `sequence_norm_bwd` |
| Timestep Norm | `ops/timestep_norm.h/cc` | `timestep_norm_fwd` | `timestep_norm_bwd` |

Each operation follows a consistent naming convention with `_fwd` and `_bwd` suffixes for forward and backward passes respectively. This pattern enables automatic gradient computation through PyTorch's autograd system.

Sources: [megalodon_extension.cc 3-9](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L3-L9)[megalodon_extension.cc 17-23](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L17-L23)

* * *

Layer 3: CUDA Kernel Layer
--------------------------

The CUDA kernel layer contains the actual computational implementations. Each operation has both CPU and CUDA implementations to ensure portability while maximizing performance on GPU hardware.

### Dual Implementation Strategy

**Runtime Device Selection and Type Dispatching**

Each operation implementation performs runtime device checking to select the appropriate execution path. The `AT_DISPATCH_FLOATING_TYPES_AND2` macro enables type-generic code that works with `float`, `double`, `at::Half`, and `at::BFloat16` data types.

### CUDA Implementation Components

The CUDA implementations leverage several specialized components:

| Component | Location | Purpose |
| --- | --- | --- |
| Custom kernels | `*_kernel.cu` files | Hand-optimized CUDA kernels for specific operations |
| cuBLAS wrapper | `blas.cc/h` | Templated GEMM operations |
| CUDA utilities | `cuda_utils.cuh` | Warp operations, thread configuration |
| Complex math | `complex_utils.cuh` | Device-side complex number operations |
| FFT kernels | `fft.cuh` | Block-level FFT implementations |

Sources: Derived from high-level diagrams

* * *

Integration and Data Flow
-------------------------

The following diagram illustrates how data flows through the architecture during a typical forward pass:

**Typical Operation Execution Sequence**

This flow ensures that:

1.   Python code remains simple and device-agnostic
2.   Type and device selection happens at runtime
3.   Performance-critical code executes on GPU when available
4.   CPU fallback ensures portability

Sources: Derived from architecture analysis

* * *

Runtime Dispatch Mechanism
--------------------------

The runtime dispatch mechanism is central to the architecture's flexibility. It operates on two dimensions: device type (CPU vs CUDA) and data type (float, half, bfloat16).

### Device Dispatch

Operations check the device type of input tensors at runtime:

**Device-Based Runtime Dispatch**

### Type Dispatch

Within each device path, PyTorch's `AT_DISPATCH_FLOATING_TYPES_AND2` macro enables compile-time template instantiation for multiple data types:

**Type Dispatching for Multiple Precision Levels**

This dual-dispatch system ensures that the same Python API works seamlessly across different devices and data types without requiring manual type conversion or device placement by the user.

Sources: Derived from architecture analysis and high-level diagrams

* * *

Operation Structure Pattern
---------------------------

All operations in the C++ layer follow a consistent structural pattern:

| Component | Purpose | Example |
| --- | --- | --- |
| Header file (`*.h`) | Function declarations and interface | `ops/ema_hidden.h` |
| Implementation file (`*.cc`) | Device dispatch and CPU implementation | `ops/ema_hidden.cc` |
| CUDA kernel file (`*.cu`) | GPU implementation | `ema_hidden_kernel.cu` |
| Define function | PyBind11 registration | `DefineEMAHiddenOp` |

Each operation exposes:

*   A forward pass function (`*_fwd`)
*   A backward pass function (`*_bwd`)
*   Both functions registered with the PyBind11 module

This consistent pattern simplifies development, testing, and maintenance of custom operations.

Sources: [megalodon_extension.cc 3-9](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L3-L9)[megalodon_extension.cc 17-23](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L17-L23)

* * *

Build System Integration
------------------------

The architecture is built as a PyTorch C++ extension using the following components:

1.   **PyBind11**: Provides Python bindings for C++ code
2.   **PyTorch Extension API**: Enables seamless tensor operations
3.   **CUDA Toolkit**: Compiles CUDA kernels
4.   **cuBLAS**: Provides optimized BLAS operations

The extension is compiled and installed via `pip install -e .`, making it available as `import megalodon_extension` in Python code.

Sources: [README.md 56-62](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md#L56-L62)

* * *

Summary
-------

The three-tier architecture provides:

*   **Clean separation of concerns**: Python interface, C++ operations, CUDA kernels
*   **Dual implementation strategy**: CPU and CUDA paths for every operation
*   **Type flexibility**: Support for float, double, half, and bfloat16 precision
*   **Consistent patterns**: All operations follow the same structural conventions
*   **Performance**: GPU acceleration when available, CPU fallback for portability
*   **Maintainability**: Modular design enables independent development of operations

This architecture enables the Megalodon model to achieve high performance while maintaining a simple Python API for users.

Sources: [megalodon_extension.cc 1-27](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/megalodon_extension.cc#L1-L27)[README.md 1-172](https://github.com/XuezheMax/megalodon/blob/cff8ba5f/README.md#L1-L172)
