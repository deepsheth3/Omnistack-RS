# OmniStack-RS — ARCHITECTURE MANIFEST v1.0
# IMMUTABLE ARCHITECTURE BIBLE — Version with a commit hash before any breaking change.
# Paste this file at the start of any Claude session to restore full context instantly.

---

## THE VISION

OmniStack-RS is a **Federated Personalization Engine** for entertainment recommendations at 100M-user scale.

**The Analogy**: Apple ships a massive server-side LLM + tiny per-iPhone weights that learn app usage patterns and pre-load apps before the user taps them. We do this for Netflix: one giant **Master LLM** understands cinema, tropes, genre psychology, and narrative structure. Each user has a **Shadow** — a small LoRA adapter (10–100 MB) trained on their private viewing history (pauses, skips, rewinds, rewatches). Master + Shadow merge at inference time to become a **personalized expert**.

**Why this beats collaborative filtering**: Standard Netflix uses "users like you liked X" (demographic clustering). We use **Semantic Intent Forecasting**: the model knows your specific current mood, recent viewing arc, and taste manifold — not just your age bracket.

**The Economic Thesis**:
| Metric | Standard LLM API | OmniStack-RS |
|--------|-----------------|--------------|
| Cost per recommendation session | $0.50 | $0.01 |
| Concurrent users per H100 | ~800 | ~10,000+ |
| Mechanism | Full BF16 KV cache | 12.8× VRAM reduction |

---

## THE 6-STAGE PERSONALIZATION FIREWALL

Every recommendation passes through exactly these 6 stages. This ordering is **non-negotiable**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Data Capture         (User Device / Private Enclave)           │
│  Stage 2: Shadow Training      (Device / Private Enclave — offline)      │
│  Stage 3: Federated Sync       (Secure Channel — Δ-weights only)         │
│  Stage 4: Manifold Pruning     (Server — 128-dim → 32-dim)               │
│  Stage 5: KV Cache Compression (Server — BF16 → 5-bit INT4+QJL)         │
│  Stage 6: Fused Inference      (Server H100/A100 — Master+Shadow merge)  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Data Capture (User Device / Private Enclave)
**Input**: Raw user interactions with the viewing interface
**Signals captured**:
- `watch_fraction`: 0.0 = skipped immediately, 1.0 = watched to end, > 1.0 = rewatched
- `pause_count`: proxy for cognitive load / emotional response
- `rewind_count`: proxy for scenes they wanted to re-experience
- `timestamp`: enables temporal decay (recent taste > old taste)
- `genre_vector`: multi-hot genre encoding (15 genres in canonical implementation)

**Privacy guarantee**: Raw viewing data **never leaves the device**. Only gradient updates (Stage 3) are transmitted.

**Implementation**: `data/synthetic/viewing_history.py` — `ViewingEvent` dataclass, `generate_user_history()`, `history_to_embedding()` with **30-day half-life temporal decay**.

### Stage 2 — Shadow Training (Device / Private Enclave)
**Input**: Viewing event log from Stage 1
**Process**: Fine-tune a per-user LoRA adapter on top of the frozen Master LLM using next-item prediction loss (given history, predict the next movie the user will enjoy).
**Output**: Shadow LoRA adapter — matrices A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, rank r=16, injected into Q and V projections only.
**Size**: ~10–100 MB per user (vs. 140 GB for Llama-3 70B)
**Implementation**: `omnistack_rs/shadow/lora.py`, `omnistack_rs/shadow/trainer.py`

### Stage 3 — Federated Sync (Secure Channel)
**Input**: Shadow LoRA delta-weights ΔW = BA
**Process**: FedAvg aggregation — server merges delta updates from N users, weighted by viewing activity volume. Differential privacy noise applied before upload (future phase).
**Privacy**: Raw viewing data is never transmitted. Only the compressed LoRA deltas.
**Implementation**: `omnistack_rs/shadow/federated.py` — `FederatedAggregator`

### Stage 4 — Manifold Pruning (Server)
**Input**: 128-dim user embedding derived from Shadow's KV cache
**Process**: Project onto Gr(32, 128) Grassmann manifold via truncated SVD basis. Discard the 96 "irrelevant" dimensions (Python syntax knowledge, physics equations, etc.).
**Output**: 32-dim taste vector that lives in the "Entertainment & Personality" subspace
**Memory impact**: 128 → 32 dims = **4× reduction** in user context storage
**Implementation**: `omnistack_rs/manifold/grassmannian.py` — `GrassmannianProjector` with `explained_variance_ratio`

### Stage 5 — KV Cache Compression (Server)
**Input**: Full BF16 attention KV cache from the attention layers
**Process (in order)**:
1. **Hadamard rotation** (applied to Q once; K/V stored permanently in rotated space)
   — WHT is orthogonal: H(q)·H(k) = q·k, so we never rotate inside the attention loop
2. **INT4 quantization**: 4-bit Lloyd-Max codebook (16 centroids), calibrated on real activations
   — Packing: two codes per byte, `byte = (code_a & 0xF) | ((code_b & 0xF) << 4)`
   — Unpacking: O(1), `even = byte & 0xF`, `odd = byte >> 4` — no branching
3. **1-bit Rademacher QJL residual**: on-the-fly PRNG (seeded, deterministic), zero SRAM overhead
   — Reconstruction: `residual ≈ sqrt(π/2) / m × ‖r‖ × Σ(sign_j × r_j)`
4. **Async eviction** via `DoubleBufferCompressor` on a background CUDA stream — decode path never blocks

**Result**: 5 bits/element vs 16-bit BF16 = **3.2× compression from quantization**
**Combined with Stage 4**: 4× × 3.2× = **12.8× total VRAM reduction**
**Implementation**: `omnistack_rs/kernels/quantize.py`, `omnistack_rs/kernels/hadamard.py`, `omnistack_rs/cache/double_buffer.py`

### Stage 6 — Fused Inference (Server — H100/A100)
**Input**: Rotated+quantized KV cache, user Shadow LoRA
**Process**:
- Shadow weights merged into Master: `W_eff = W_base + (α/r) × B@A`
- Fused Triton attention kernel executes:
  `TMA async load → INT4 dequant → Rademacher QJL reconstruct → QK^T (FP32) → online softmax → BF16 output`
- Online softmax uses running max/sum (Flash Attention pattern) — **never materializes full attention matrix**
- Softmax always in FP32 — BF16 input upcast before `exp()`, downcast after normalization

**Output**: Ranked list of personalized movie/TV recommendations with confidence scores
**Implementation**: `omnistack_rs/kernels/fused_attention.py`

---

## MATHEMATICAL GOALS

### Primary KPI: 12.8× VRAM Reduction

| Compression Stage | Mechanism | Factor |
|------------------|-----------|--------|
| Manifold Pruning (Stage 4) | 128-dim → 32-dim | **4.0×** |
| INT4 Quantization (Stage 5) | BF16 (16-bit) → 4-bit | **4.0×** |
| QJL Residual overhead | +1 bit for residual | **−0.8×** net → 3.2× |
| **Combined (Stage 4 × Stage 5)** | **4.0× × 3.2×** | **12.8×** |

At 12.8× VRAM reduction: a single H100 (80 GB HBM3) serving 800 concurrent users in standard BF16 can serve **~10,000 concurrent personalized users** with OmniStack-RS.

### Perplexity Budget
- **Target**: 5-bit (INT4+QJL) perplexity within **0.3 nats** of 16-bit BF16 baseline
- **Measured on**: movie description next-token prediction (Phi-2 or Llama-3-8B)
- **Mechanism**: Hadamard rotation sphericizes outliers → Gaussian distribution → Lloyd-Max centroids are optimal for Gaussian → QJL residual recovers truncation error

### Throughput Target
- Fused Triton kernel: ≥ **80% of H100 theoretical peak TFLOPS**
- vs PyTorch `F.scaled_dot_product_attention`: ≥ **2× throughput** at seq_len=4096
- No WHT in inner loop (key fix) — only INT4 dequant + Rademacher PRNG + QK^T + softmax

### Context Window
- Handle **100K-token user history** (~5 years of viewing) in **25% of baseline VRAM**
- Sliding window: last 1,024 tokens in full BF16 for immediate context
- Historical tokens: compressed to 5-bit, accessed via paged KV cache

### Manifold Dimensionality
- **Demo target**: 8-dim Grassmannian separates 5 user personas with ARI > 0.90
- **Production target**: 32-dim captures ≥ 95% of variance in entertainment embeddings
- **Validation**: `GrassmannianProjector.explained_variance_ratio` and `find_rank_for_variance(target=0.95)`

---

## HARDWARE CONSTRAINTS

| Environment | Hardware | Constraints |
|-------------|----------|-------------|
| Dev (Mac) | CPU only, Python 3.13.3, PyTorch 2.7.0 | No CUDA, no Triton GPU kernels |
| GPU CI | Modal H100 (80 GB HBM3) | Full Triton, TMA (arch ≥ 90), mandatory gate |
| Production | 8× H100 via InfiniBand NVLink | ZeRO-3 + Tensor Parallelism |
| Fallback GPU | A100 (40/80 GB) | TMA not available — software-pipelining fallback |

### CPU Dev Rules (Mac)
- All unit tests **MUST pass** under `TRITON_INTERPRET=1`
- TMA (`tl._experimental_descriptor_load`) **MUST be stubbed** via `conftest.py` `TMAStub`
- Phase 0 demo scripts (`demo/manifold_demo.py`, `demo/compression_fidelity.py`) run on CPU-only PyTorch

### GPU Production Rules
- TMA gated on: `triton.runtime.driver.active.get_current_target().arch >= 90`
- A100 fallback: `tl.load(..., eviction_policy="evict_first")` for streaming behavior
- BF16 matmul: `torch >= 2.3` required for numerical stability guarantee
- Triton: pin to `>= 3.0.0` for `tl.make_tensor_descriptor` TMA API

---

## INVARIANTS (NEVER VIOLATE)

These were discovered via Senior Code Review and saved us from production disasters.

**1. WHT in the inner attention loop is FORBIDDEN.**
Q is rotated once before the loop via `rotate_queries()`. K/V are stored permanently in rotated space via `rotate_kv_cache()` (called at prefill). The `_fused_attention_kernel` never calls any WHT function. Violation causes register spill → VRAM round-trip → defeats TMA purpose.

**2. 3-bit packing is FORBIDDEN. Use INT4 (nibble) only.**
All quantization uses 4-bit alignment. Pack: `byte = (code_a & 0xF) | ((code_b & 0xF) << 4)`. Unpack: `even = byte & 0xF`, `odd = (byte >> 4) & 0xF`. This is O(1) and branch-free. Violation causes variable-shift ALU overhead that negates bandwidth savings.

**3. Stored G matrices in SRAM are FORBIDDEN.**
All QJL projections use on-the-fly Rademacher PRNG (seeded, deterministic) generated inside the Triton kernel. There is no `G_ptr` parameter in the fused kernel. Violation steals 32 KB of SRAM from attention accumulators, reducing GPU occupancy.

**4. Synchronous eviction inside the decode loop is FORBIDDEN.**
All eviction and KV compression happens in `DoubleBufferCompressor.compress_stream` — a background CUDA stream. The decode stream queries events non-blockingly with `.query()`. Violation causes "Time-Between-Tokens" jitter (stuttering) correlated with sliding window advances.

**5. TMA cannot be unit-tested with TRITON_INTERPRET=1.**
All TMA calls are wrapped in `try_tma_load()` with a `TMAStub` fallback. CPU tests inject the stub via `conftest.py` monkeypatch. The Modal H100 CI script is the **mandatory gate** for Phase 4+ — no merging without passing remote GPU tests.

**6. Softmax must always be computed in FP32.**
BF16 input → upcast to FP32 → `tl.exp()` → online normalization → downcast to BF16 output. Never compute `tl.exp()` on BF16 values. Violation causes probability distribution collapse at long contexts (> 4K tokens).

**7. The 12.8× VRAM reduction is the primary KPI.**
Every architectural decision is evaluated against this target. If a change reduces VRAM savings, it requires explicit justification and quantified trade-off.

---

## DIRECTORY STRUCTURE (CANONICAL)

```
Omnistack_RS/
├── MANIFEST.md                  ← THIS FILE — Architecture Bible
├── pyproject.toml               ← Pinned deps: triton>=3.0.0, torch>=2.3
├── .gitignore
├── ci/
│   └── run_gpu_tests.py         ← Modal H100 CI — mandatory Phase 4+ gate
├── omnistack_rs/
│   ├── config.py                ← OmniConfig dataclass (single source of truth)
│   ├── kernels/
│   │   ├── hadamard.py          ← WHT butterfly; rotate_queries(), rotate_kv_cache()
│   │   ├── quantize.py          ← INT4 nibble pack + Rademacher QJL encode
│   │   ├── dequantize.py        ← INT4 + QJL reconstruct
│   │   ├── tma_utils.py         ← TMA descriptors + TMAStub (CPU fallback)
│   │   └── fused_attention.py   ← MAIN kernel — Stage 6 core, no WHT inner loop
│   ├── quantization/
│   │   ├── codebook.py          ← Lloyd-Max 4-bit calibration (16 centroids)
│   │   ├── qjl.py               ← 1-bit Rademacher QJL (CPU reference path)
│   │   └── polar_quant.py       ← PolarQuantKV orchestration + EncodedKV
│   ├── attention/
│   │   ├── reference.py         ← Pure-PyTorch GQA + FP32 softmax (ground truth)
│   │   ├── gqa.py               ← GQA head mapping, repeat_kv utility
│   │   └── backend.py           ← AttentionDispatcher (fused vs. reference)
│   ├── cache/
│   │   ├── kv_cache.py          ← PagedKVCache block allocator
│   │   ├── eviction.py          ← SlidingWindowEviction + LatentSummaryEviction
│   │   ├── double_buffer.py     ← Async DoubleBufferCompressor (Stage 5 critical path)
│   │   └── paged.py             ← KVCacheBlock dataclass, block pool
│   ├── manifold/
│   │   ├── grassmannian.py      ← GrassmannianProjector (Stage 4) + SVDProjectionResult
│   │   └── pruning.py           ← ManifoldPruner (angular dedup + norm filter)
│   ├── shadow/
│   │   ├── lora.py              ← ShadowLoRA (Stage 2 LoRA adapter)
│   │   ├── trainer.py           ← ShadowTrainer (Stage 2 fine-tuning)
│   │   └── federated.py         ← FederatedAggregator (Stage 3 FedAvg)
│   └── distributed/
│       ├── comm.py              ← all_reduce, all_gather, scatter wrappers
│       ├── tensor_parallel.py   ← ColumnParallelAttention (head sharding)
│       └── zero3.py             ← DeepSpeed ZeRO-3 wrap_zero3()
├── data/
│   └── synthetic/
│       └── viewing_history.py   ← ViewingEvent, generate_user_history(), history_to_embedding()
├── demo/
│   ├── manifold_demo.py         ← Phase 0b: Grassmannian persona clustering
│   └── compression_fidelity.py  ← Phase 0c: Perplexity vs bit-width benchmark
├── tests/
│   ├── conftest.py              ← fixtures + TMAStub injection
│   ├── unit/
│   └── integration/
├── benchmarks/
│   ├── bench_attention.py       ← Triton vs PyTorch SDPA throughput
│   ├── bench_compression.py     ← Perplexity vs bit-width sweep
│   └── bench_context_window.py  ← 100K token VRAM stress test
└── scripts/
    ├── calibrate_codebook.py
    ├── profile_kernel.py
    └── generate_rotated_kv.py   ← Pre-rotate KV checkpoint at load time
```

---

## THE 3 CHARTS THAT CLOSE THE DEAL

For a Netflix or Google engineer reviewing this repository, three visualizations are the proof:

**Chart 1 — Compression Fidelity** (`benchmarks/bench_compression.py`)
- X-axis: effective bit-width {FP32=32, BF16=16, INT8=8, INT4=4, INT4+QJL=5}
- Y-axis: perplexity on movie description prompts (lower = better)
- Target: INT4+QJL within **0.3 nats** of BF16
- Tagline: "5-bit matches 16-bit quality. 3.2× the users per GPU."

**Chart 2 — Context Window VRAM Stress Test** (`benchmarks/bench_context_window.py`)
- X-axis: context length {1K, 10K, 50K, 100K tokens}
- Y-axis: peak VRAM GB (OmniStack-RS vs PyTorch baseline)
- Target: 100K tokens at **25% of baseline VRAM** (12.8× improvement)
- Tagline: "Handle a user's 5-year viewing history in a single GPU pass."

**Chart 3 — Grassmannian Persona Clustering** (`demo/manifold_demo.py`)
- 2D t-SNE scatter of 500 synthetic users × 5 personas in 8-dim Grassmannian space
- Target: visually separable clusters, **ARI > 0.90**
- Caption: "8-dimensional manifold perfectly separates 5 user taste profiles. 128→8 dims = 16× compression with zero persona information lost."

---

## QUICK-START FOR NEW SESSIONS

```bash
# Clone and install
cd /Users/deepsheth/Documents/Projects/Omnistack_RS
pip install -e ".[dev,demo]"

# Run Phase 0 demo (Mac CPU — no GPU required)
python demo/manifold_demo.py         # Grassmannian clustering, ARI > 0.90
python demo/compression_fidelity.py  # Perplexity vs bit-width on Phi-2

# CPU unit tests (all phases using Triton interpreter)
TRITON_INTERPRET=1 pytest tests/unit/ -v

# GPU tests (mandatory gate for Phase 4+)
python ci/run_gpu_tests.py
```

---

*OmniStack-RS MANIFEST v1.0 — The Master & The Shadow — 2026-04-23*
