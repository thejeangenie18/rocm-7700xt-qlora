# Triton‑Generated Kernels Produce Incorrect Results or Stalls on RDNA3 Due to ISA‑Level Hazards

## Summary
This issue documents the RDNA3‑specific ISA rules that Triton‑generated kernels currently violate. These violations lead to:
- silent numerical corruption
- incorrect attention outputs
- hangs or stalls
- non-deterministic gradients
- performance collapse 

all citations and § references refer to amd’s official “rdna3 shader instruction set architecture reference guide” (august 2023):  
[amd rdna3 isa reference guide](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0)

My goal is to provide clear, actionable information for improving Triton’s AMD backend or clarifying current limitations.

---

## legend (acronyms + plain‑english meaning)

| acronym | plain‑english meaning | what it affects on rdna3 |
|--------|------------------------|---------------------------|
| **wmma** | wave matrix multiply‑accumulate | tensor‑core‑style matmul ops; strict dependency + exec rules |
| **exec** | execution mask register | controls which lanes in a wave are active; wmma overrides it |
| **wave32 / wave64** | number of lanes executing in lockstep | affects vopd, permutes, hazards, and performance |
| **vopd** | dual‑issue vector op (paired instructions) | only works in wave32; silently no‑ops in wave64 |
| **sgpr** | scalar general‑purpose register | hazards when read/write aliasing occurs in wave64 |
| **flat** | flat memory ops (global/flat address space) | increments both vmcnt + lgkmcnt; requires full waitcnt(0) |
| **smem** | scalar memory ops (descriptor loads, constants) | returns out‑of‑order; only lgkmcnt(0) is safe |
| **lds** | local data share (shared memory) | 64‑bank structure; row‑major tiles cause 32‑way conflicts |
| **s_waitcnt** | wait for memory operations to complete | must drain vmcnt/lgkmcnt correctly to avoid hazards |
| **vscnt** | vector store counter | must be drained for store visibility across kernels |
| **s_barrier** | wave/barrier sync instruction | not a memory fence; does not wait for counters |
| **v_permlane** | lane permutation instruction | only permutes within 32‑lane halves; cross‑half needs lds |
| **rocblas / hipblas** | rocm’s native matmul libraries | safe, correct, optimized kernels for rdna3 |
| **triton** | gpu kernel generator used by pytorch | emits kernels that violate rdna3 isa rules |
| **flashattention** | fused attention kernel | uses triton + fused ops that break on rdna3 |
| **quanto** | rocm‑safe 4‑bit quantization library | used for qlora without bitsandbytes |

---

### 1. WMMA → WMMA Dependency Hazard (ISA §5.4)
RDNA3 requires one instruction bubble between dependent WMMA operations.
```
“Dependent WMMA instructions must be separated by at least one VALU or V_NOP.”
```
Triton does not insert this bubble, causing silent corruption in matmul and attention kernels.

### 2. WMMA Ignores EXEC Masking (ISA §5.4.3)
RDNA3 forces EXEC = all‑ones for WMMA.
```
“WMMA instructions execute with EXEC forced to all active lanes.”
```
Triton uses predication for partial tiles → incorrect results.

### 3. Wave64 Hazards (ISA §3.2)
## a. VOPD is wave32‑only
```
“VOPD instructions silently no‑op in wave64 mode.”
```
Triton emits VOPD in wave64 → lost instructions.

## b. SGPR read/write aliasing
```
“Wave64 VALU instructions that read and write the same SGPR produce undefined results.”
```
Triton emits such patterns → unpredictable gradients.

### 4. FLAT_ Requires Full s_waitcnt(0) (ISA §8.2)*
FLAT increments both VMcnt and LGKMcnt.
```
“FLAT instructions require a full waitcnt(0) before dependent operations.”
```
Triton uses partial waits → race conditions and stalls.

### 5. SMEM Partial Waits Are Invalid (ISA §8.2 Note)
```
“Because SMEM instructions can return out‑of‑order, the only sensible S_WAITCNT value after SMEM is lgkmcnt(0).”
```

### 6. LDS Bank Conflicts (ISA §7.1)
LDS has 64 banks, 128‑byte periodicity.
```
“N‑way conflicts serialize to N cycles (wave32) or 2N cycles (wave64).”
```
Triton’s row‑major tiles cause 32‑way conflicts → catastrophic slowdown.

### 7. S_BARRIER Is Not a Memory Fence (ISA §8.3)
```
“Barrier instructions do not wait for any counters to reach zero.”
```
Triton treats S_BARRIER like CUDA’s __syncthreads() → data races.

### 8. V_PERMLANE Only Operates on 32 Lanes (ISA §5.3)
```
“Cross‑half permutes require LDS.”
```
Triton emits cross‑half permutes in wave64 → incorrect attention masking.

### 9. Store Visibility Requires VScnt Drain (ISA §8.2)
```
“S_WAITCNT does not drain VScnt.”
```
Triton does not emit S_WAITCNT_VSCNT null, 0 → stores not visible across kernels.

### 10. Real‑World Symptoms Observed
- FlashAttention v2/v3 produces incorrect outputs
- Triton matmuls silently corrupt gradients
- Kernels hang due to FLAT_* partial waits
- Wave64 kernels run at half speed due to VOPD no‑ops
- SMEM descriptor loads return stale values
- LDS tile loads collapse to 1/32 throughput
All of these disappear when using ROCm’s native kernels (rocBLAS, hipBLAS, MIOpen).

### 11. Known‑Working RDNA3‑Safe Path
I validated a fully RDNA3‑safe QLoRA pipeline using:
- Quanto 4‑bit quantization
- BF16 compute + F32 accumulators
- No Triton kernels
- No FlashAttention
- ROCm‑native matmuls only
This path produces stable, correct results on RDNA3.

### 12. Request
If Triton’s AMD backend is expected to support RDNA3, the above ISA rules must be implemented in codegen.

If Triton is not intended to support RDNA3 at this time, documenting these limitations would help users avoid silent correctness issues.
