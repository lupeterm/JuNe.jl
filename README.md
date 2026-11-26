# JuNe.jl
JuNe: Julia matrix assembly embedded into [NeoN](https://github.com/exasim-project/NeoN)

## Changelog

### 0.2.0 (2026-11-26)

- **Batched face-based strategy**: the faces in batches without shared cells, so the interior faces
  need no atomics; on the CPU, on the GPU with KernelAbstractions (`assemble_batched_gpu`) and with
  CUDA.jl kernels (`assemble_batched_cuda`).
- **Matrix value types**: one code path assembles NeoN's coupled `Vec3` matrix and the segregated
  `Scalar` one (`VectorType`).
- **Any floating-point type on the GPU**: the entry points take a keyword `FT` (one type, or one per
  array), every operator computes in its own precision and the kernels accumulate in the matrix type.
- **Relaxed atomics**: matrix and rhs updates use a relaxed (monotonic), device-scope atomic add
  instead of Atomix's, which is acquire-release on CUDA; fp64 and fp32 are emitted as PTX `red` on
  the GPU. Face-based GPU assembly of div + laplacian at 400^3: 16.55 ms -> 13.28 ms. The PTX makes
  the GPU path NVIDIA-specific.
- **Operators alone**: the GPU kernels read the inputs of absent operators as zero, so div or
  laplacian can be assembled on their own.
- **`benchmark_precision`**: the mixed-precision study of the GPU assembly (fp64, fp32, fp16, bf16:
  whole, one operator reduced, each operator alone), with allocation, conversion, back-conversion and
  reset timed separately, an NVTX range per configuration for profilers and `PRECISION_CONFIGS` to
  select configurations; an opt-in fp8 (E4M3) storage configuration (`Float8E4M3`).
- Test suite (`test/runtests.jl`); new dependencies Adapt, Printf, UnsafeAtomics and NVTX.

### 0.1.0

Face-based, global face-based and cell-based assembly of ddt, div and laplacian into NeoN's linear
system, serial, multi-threaded and on the GPU (KernelAbstractions).
