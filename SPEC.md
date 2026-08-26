# Native 1-Bit Instruction Model Research Specification
**Version 2 — Updated August 2026**

> This document supersedes the original ChatGPT-generated draft. All prior-art references, performance numbers, and inference benchmarks have been updated to reflect the state of the field as of mid-2026. The project's hard constraints and design philosophy are unchanged.

---

## 0. Project Intent

Build a genuinely native 1-bit instruction-following language model from random initialization, using Python throughout the project, with PyTorch and NumPy as the only non-standard computational dependencies.

The model must not be trained as a conventional floating-point language model and quantized afterward. Its forward-pass weights must be binary from the first training step and remain binary throughout training and inference.

The project will use Bonsai 8B as a behavioral teacher and reference model. Bonsai is not the source of the student's weights. The student has its own architecture, tokenizer, initialization, packed weight representation, training procedure, and checkpoint format.

The second major goal is a deliberately simple and hardware-oriented weight file format, `model.weights`, designed for direct binary storage, predictable offsets, alignment, memory mapping, sequential access, and eventual custom kernels. Power-of-two dimensions, group sizes, tensor alignment, and storage blocks should be used wherever they improve regularity or implementation simplicity.

The project should be treated as a research implementation, not a wrapper around existing LLM libraries.

---

## 1. Hard Constraints

These constraints are mandatory unless an experiment explicitly records a deviation.

### 1.1 Language

Use Python for the entire project.

Do not move training to Rust, Go, C++, JAX, or another language.

### 1.2 Core Numerical Dependencies

Allowed:

- Python standard library
- PyTorch
- NumPy

Do not make the project depend on:

- Hugging Face Transformers
- Hugging Face Accelerate
- bitsandbytes
- optimum
- vLLM
- llama.cpp
- MLX
- JAX
- TensorFlow
- Keras
- third-party quantization libraries
- third-party model implementations
- third-party optimizer implementations
- third-party tokenizer implementations
- third-party checkpoint formats as the canonical project format

The reason is not ideological. The project is specifically intended to expose and control the low-level implementation details of binary weights, quantization, gradient estimation, tensor layout, serialization, and training behavior.

Third-party packages may be used temporarily for an isolated research comparison only if the comparison is explicitly marked as external and the package is not part of the core implementation.

### 1.3 Model Implementation

Implement the Transformer manually in PyTorch.

Do not import a pretrained Transformer architecture from another package.

Implement manually:

- token embedding
- positional encoding / RoPE
- RMSNorm or SubLN normalization
- attention
- causal masking
- GQA if used
- MLP / SwiGLU if used
- binary linear layers (BitLinear equivalent)
- output head
- loss calculation
- optimizer or optimizer wrapper where practical
- checkpoint serialization
- inference loading
- sampling

### 1.4 Native 1-Bit Requirement

The effective weights used by every binary weight-bearing layer must be binary from initialization.

The canonical binary domain is:

- `False` / bit 0
- `True` / bit 1

For mathematical computation, interpret the bits as:

- `False → -1`
- `True  → +1`

Do not interpret `False` as numerical zero for the primary binary linear algebra path.

The file representation must pack bits densely. A Python list of `bool` values is only a logical abstraction and must never be the serialized representation.

### 1.5 Full Model Coverage

The research target is end-to-end binary weights.

The binary representation should eventually cover:

- token embedding weights
- attention projections
- MLP projections
- final LM head

The following may remain higher precision unless an experiment explicitly tests binary alternatives:

- activations (see Section 8 for the activation quantization roadmap)
- RMSNorm / SubLN statistics and parameters
- RoPE calculations
- softmax
- loss computation
- optimizer state
- temporary training buffers

This distinction is important. "1-bit model" refers to model weight representation, not the requirement that every numerical quantity in training be one bit.

### 1.6 No Post-Training Conversion

A conventional FP32/BF16 model may not be trained and then converted into the primary 1-bit model.

Floating-point auxiliary state is permitted only as an optimization mechanism for discrete parameters, and the exact role of that state must be documented. The model's effective forward weights remain binary at every step.

### 1.7 Power-of-Two Design

Where dimensions or storage structures are under our control, prefer powers of two.

Examples:

- vocabulary size
- hidden dimension
- head dimension
- number of heads
- number of KV heads
- MLP dimension
- binary quantization group size
- tensor alignment
- file block size
- metadata block size

Power-of-two choices are not required where they provide no meaningful advantage, but arbitrary dimensions should not be introduced without reason.

---

## 2. Important Distinction: Python bool vs Packed Bits

A Python `bool` object is not a one-bit storage type. Do not store weights as Python objects.

The conceptual interface is:

```python
True
False
```

The physical representation is packed bits, for example:

```text
10110100 01101001 ...
```

For 128 binary weights:

```text
128 weights
= 128 bits
= 16 bytes
```

The canonical on-disk weight representation should therefore be a byte-addressable packed-bit array with explicit metadata describing shape, bit order, and any associated scale representation.

The first implementation should use NumPy or PyTorch integer tensors for packing and unpacking. Do not prematurely hide the operation behind a third-party bit tensor library.

---

## 3. Core Research Questions

### Primary Question

> Can a Transformer designed around binary weights, power-of-two tensor geometry, and a binary-native training procedure learn useful language representations and instruction-following behavior from random initialization, while retaining an efficient packed on-disk representation suitable for direct memory mapping and eventual bitwise kernels?

### Secondary Question

> Can Bonsai 8B act as a behavioral teacher that substantially improves a native 1-bit student's instruction-following quality without transferring or reconstructing Bonsai's weights?

### Third Question

> How much model quality can be obtained per byte of model weight storage when the architecture and storage format are designed around binary weights from the beginning?

### Fourth Question (Added 2026)

> Does autoregressive distillation from a strong teacher (as demonstrated by FBI-LLM) provide a more stable training signal than next-token prediction alone for native binary training from scratch? How does this interact with the STE-based optimization path?

---

## 4. Relationship to Prior Work and the Current Landscape

### 4.1 The BitNet Family (Microsoft Research)

The most directly relevant training-side work comes from Microsoft Research's BitNet series.

**BitNet (Wang et al., 2023)** introduced BitLinear as a drop-in replacement for `nn.Linear` enabling 1-bit weight training from scratch. It uses AbsMean quantization for weights and AbsMax per-token quantization for activations, paired with SubLN normalization for training stability. The original BitNet uses binary weights `{-1, +1}`.

**BitNet b1.58 (Ma et al., 2024)** extended this to ternary weights `{-1, 0, +1}` — 1.58 bits by information theory. At 3B parameters it matches FP16 LLaMA on perplexity while using 3.55× less memory and running 2.71× faster. A 2B model trained on 4T tokens was released under MIT license in April 2025, making it the first open-weight natively trained 1-bit LLM at useful scale.

**BitNet a4.8 (Wang et al., 2024)** introduced hybrid quantization and sparsification for activations: 4-bit activations for attention and FFN inputs, 8-bit with top-K sparsification for intermediate states. It activates only 55% of parameters per forward pass, supports 3-bit KV cache, and matches BitNet b1.58 quality with lower inference cost.

**BitNet v2 (Wang et al., 2025)** introduced `H-BitLinear` — applying an online Hadamard transformation before activation quantization to smooth outlier-dominated activation distributions into near-Gaussian forms amenable to 4-bit quantization. This trains from scratch with 8-bit activations matching b1.58, then continues training with native 4-bit activations. This is the current state of the art for activation quantization in native 1-bit models.

**bitnet.cpp** is Microsoft's official inference framework for 1-bit LLMs, with optimized I2_S, TL1, and TL2 kernels for ARM and x86. ARM speedup: 1.37×–5.07×. x86 speedup: 2.37×–6.17×. Energy reduction: 55%–82%. The January 2026 update added a 1.15×–2.1× further speedup through parallel kernel implementations with configurable tiling. GPU inference kernels were added in May 2025.

The key design decisions in the BitNet approach:
- **AbsMean scaling** for weights at ≤2 bits (outperforms AbsMax at very low bit-widths; AbsMax is better at ≥3 bits)
- **AbsMax per-token** for 8-bit activations
- **SubLN normalization** (pre-normalization variant of LayerNorm placed before attention and FFN) for training stability, enabling larger learning rates than Pre-LN

This project targets stricter binary `{-1, +1}` weights (no zero). The above BitNet work is directly relevant to the optimization mechanism but the pure-binary commitment means the scaled-binary family is a comparison point, not the target.

### 4.2 FBI-LLM: Fully Binarized Training from Scratch

FBI-LLM (Ma, Sun, Shen, 2024) is the single most relevant prior work to this project. It demonstrated for the first time that a truly binary model (not ternary like b1.58) can be trained from scratch at useful scale: 130M, 1.3B, and 7B parameters, achieving competitive results against full-precision counterparts.

The key contribution is **autoregressive distillation (AD) loss**: rather than pure next-token cross-entropy, the student is trained to match the token probability distribution of a stronger full-precision teacher model using a distillation objective. The training corpus used was 1.26T tokens (Amber dataset).

FBI-LLM found:
- Pretrained full-precision initialization is NOT necessary for binarized LLMs trained from scratch
- Autoregressive distillation substantially stabilizes binary training versus pure NTP
- The flip-flop ratio (a measure of weight oscillation) is informative for diagnosing optimization stability

This is a critical reference for the student's training pipeline. The AD loss should be considered as a first-class training objective, not a later experiment, because it directly addresses the training instability that is the biggest challenge in pure binary training.

### 4.3 Bonsai (PrismML) — Updated Status

**Bonsai 8B** (PrismML, released March 31, 2026) is a true end-to-end 1-bit model. All weights — embeddings, attention, MLP, LM head — are `{-1, +1}` with no higher-precision escape hatches. The effective storage is 1.125 bits per weight (1 sign bit per weight + 1 FP16 scale per group of 128 weights).

Performance benchmarks (March 2026):
- M4 Pro Mac: 131 tokens/sec
- RTX 4090: 368 tokens/sec
- iPhone 17 Pro Max: ~44 tokens/sec
- Memory footprint: 1.15 GB (vs. ~15 GB for FP16 equivalent)
- Trained on Google TPU v4

PrismML's training procedure is based on QAT with proprietary Caltech mathematics (Babak Hassibi's group). The IP is owned by Caltech and exclusively licensed to PrismML. The training recipe is not public. Do not assume Bonsai's compression algorithm is reproducible from the checkpoint.

**Bonsai 27B** (PrismML, released July 14, 2026) is a multimodal 1-bit/ternary model based on Qwen3.6-27B. Architecture is the Qwen3.6 architecture (unchanged) with end-to-end low-bit weights applied. Both 1-bit (3.9 GB, 89.5% of FP16 quality) and ternary (5.9 GB, 94.6% of FP16 quality) variants ship under Apache 2.0. This demonstrates that the native low-bit approach now scales to 27B-class models running on phones.

Key insight from Bonsai 27B: conventional sub-4-bit post-training quantization collapses selectively on harder benchmarks (AIME, LiveCodeBench, agentic tasks) even when short-form scores remain high. Native QAT avoids this failure mode.

Bonsai remains the behavioral teacher reference and primary target for distillation. Both the 8B and 27B models are available as GGUF and MLX artifacts. However, do not assume that access to the Bonsai weights reveals the training procedure.

### 4.4 Weight Quantization: AbsMean vs AbsMax

For the binary case `{-1, +1}`, the quantization function is `sign(W)`, which maps any nonzero weight to its sign. The key design decision is the **scale factor** for the scaled-binary variant:

**AbsMean scaling** (used by BitNet b1.58):
```
α = mean(|W|)
W_binary = sign(W)
effective = α * sign(W)
```

**AbsMax scaling** (used by original BitNet for activation quantization):
```
α = max(|W|)
```

Research (including the 2025 ablation by Ma et al.) shows AbsMean significantly outperforms AbsMax for ≤2-bit weight quantization. AbsMax leads to more stable training in the activation path at 8 bits. This project should default to AbsMean for any per-group weight scales, and AbsMax per-token for activation quantization.

### 4.5 References

- BitNet (original): https://arxiv.org/abs/2310.11453
- BitNet b1.58 (JMLR 2025): https://jmlr.org/papers/v26/24-2050.html
- BitNet b1.58 2B4T Technical Report: https://arxiv.org/abs/2504.12285
- BitNet a4.8: https://arxiv.org/abs/2411.04965
- BitNet v2 (Hadamard activations): https://arxiv.org/abs/2504.18415
- FBI-LLM: https://arxiv.org/abs/2407.07093
- bitnet.cpp (inference paper): https://arxiv.org/abs/2410.16144
- Bonsai 8B announcement: https://prismml.com/news/bonsai-8b
- Bonsai 8B GGUF: https://huggingface.co/prism-ml/Bonsai-8B-gguf
- Bonsai 27B announcement: https://prismml.com/news/prismml-releases-bonsai-27b
- Microsoft BitNet repo: https://github.com/microsoft/BitNet

---

## 5. First Architectural Target

Do not begin with an 8B model.

First implement a tiny architecture whose purpose is to validate every mechanism. Then scale through several checkpoints.

Recommended progression:

```text
~50M parameters
      ↓
~125M parameters
      ↓
~350M parameters
      ↓
~1B parameters
      ↓
larger target only after the method is stable
```

A candidate 100M-ish architecture using clean powers of two:

```text
vocab_size       = 32768 or 65536
hidden_size      = 512 or 1024
head_dim         = 64 or 128
num_heads        = power of two (e.g. 8 or 16)
num_kv_heads     = power of two dividing num_heads (GQA)
mlp_size         = power of two (e.g. 2048 or 4096)
num_layers       = 8, 12, or 16
binary_group     = 128
```

The implementation must provide a small architecture calculator that prints:

- total logical parameters
- binary weight bits
- packed weight bytes
- scale bytes if scales are used
- higher-precision non-weight bytes
- estimated file size
- estimated FP16 equivalent size
- compression ratio
- estimated bytes/parameter

---

## 6. Binary Weight Representation

### 6.1 Canonical Logical Representation

Every binary weight is conceptually:

```text
0 / False / -1
1 / True  / +1
```

The primary research implementation should use a deterministic bit order. Choose one convention, document it in the file format, and never silently change it.

Recommended convention:

- bit index 0 corresponds to the least-significant bit of the first storage byte
- weights are flattened row-major before packing

### 6.2 Quantization Function

For a given weight tensor W, the binary quantization applied during the forward pass is:

```python
# For pure binary (no scale):
w_binary = sign(W)  # maps to {-1, +1}

# For scaled binary (recommended for training stability):
alpha = mean(|W|)   # AbsMean scaling factor (one per group of 128)
w_effective = alpha * sign(W)
```

The AbsMean scaling per group of 128 is empirically better than AbsMax for ≤2-bit weight quantization. It is the approach used in the production Bonsai and BitNet models.

### 6.3 Experimental Families

Run two distinct experimental families.

**A. Pure binary:**

```text
w = {-1, +1}
```

No per-group floating-point magnitude. This is the strictest interpretation of the research goal.

**B. Scaled binary:**

```text
w_i = s_g * sign(bit_i)
```

One FP16 or BF16 AbsMean scale per group of 128 weights. This is the format used by production Bonsai 8B (1.125 bits/weight effective).

Do not collapse these into one format. They are different experiments.

### 6.4 Group Size

Use a default group size of `128 = 2^7` for the scaled variant. This is directly comparable to Bonsai's published format and packs into 16 bytes.

Permit other power-of-two group sizes as ablations:

```text
32 / 64 / 128 / 256 / 512
```

---

## 7. Training Problem and Discrete Optimization

This is the most important technical problem in the project.

A true Boolean parameter is non-differentiable. Standard gradient descent cannot directly update `weight = True` using an ordinary derivative.

The implementation must choose and compare a training mechanism. As of 2026, the literature provides several viable approaches.

### 7.1 Baseline: Straight-Through Estimator (STE)

Implement the first training baseline with a continuous auxiliary parameter `w_master`, while keeping the forward-path effective parameter binary.

```python
w_binary = binary_quantize(w_master)  # sign(w_master)
w_effective = w_master + (w_binary - w_master).detach()
# forward uses w_binary, backward flows to w_master
```

This does NOT make the final model a floating-point model. It is an optimization technique for a discrete model.

**Known limitation (2025 research):** STE introduces an inherent forward/backward mismatch. This mismatch is associated with optimization instability and accuracy degradation in deeper networks. Pure STE tends to degrade with depth. Mitigation strategies include:

- Gradient clipping on the master parameters
- Learning rate warmup schedules
- Weight decay calibrated to bit-width (use AbsMean normalization)
- SubLN normalization rather than Pre-LN (empirically more stable for binarized models per the original BitNet paper)

### 7.2 Decoupled STE (Improved Baseline)

Recent work (2024) shows that separating forward-pass temperature from backward-pass gradient dispersion can significantly improve STE performance. The "Decoupled Straight-Through" (Decoupled ST) approach introduces independent temperatures for exploration (forward) and gradient spreading (backward):

```python
# Forward: use temperature tau_f for stochasticity
p = sigmoid(w_master / tau_f)
w_binary_sample = bernoulli(p)

# Backward: use temperature tau_b for gradient
# (different from tau_f)
```

This is more expensive than vanilla STE but substantially outperforms it on discrete optimization benchmarks. Consider as the primary baseline after vanilla STE is proven to work.

### 7.3 Autoregressive Distillation (FBI-LLM Method)

FBI-LLM demonstrated that using a teacher's output distribution as the training target rather than (or in addition to) ground-truth next-token labels produces significantly more stable binary training from scratch:

```text
L = CE(student_logits, ground_truth_tokens)  -- standard NTP
  + KL(student_distribution || teacher_distribution)  -- distillation
```

This is a first-class training objective for this project, not a later experiment. It directly addresses the optimization instability that makes pure binary training difficult.

The teacher does not need to be Bonsai initially. Any strong FP16 language model on the same tokenizer can serve as a distillation teacher during pretraining. Bonsai-guided distillation becomes relevant in the instruction-tuning phase.

### 7.4 Alternative Optimization Experiments

After the STE + AD-loss baseline is functioning, investigate:

- Progressive layer-wise binarization (layerwise progressive freezing): recent work (2026) shows that gradually replacing clipped weights with hard binary counterparts layer-by-layer avoids depth-scaling failures of global STE, though this approach requires careful ordering
- Stochastic bit flipping: keeps weights binary throughout training; avoids the STE mismatch entirely but requires a different gradient formulation
- Bernoulli/probabilistic parameterization: treat each weight as a Bernoulli random variable with learnable probability

Do not implement all of these initially. The baseline STE + AD-loss must work first.

### 7.5 Initialization

Initialize the underlying training state randomly, but quantize immediately before the first forward pass.

There must never be a warm-up phase where the model learns as a conventional FP model.

The first training iteration must execute using binary effective weights.

---

## 8. Activations and Normalization

Start with higher-precision activations. The goal of the first project phase is native binary weights, not simultaneous binary activations.

Use:

- BF16 or FP16 on supported accelerators for activations during large-scale training
- FP32 accumulation where numerical stability requires it

### 8.1 Normalization Choice

Use **SubLN** (sub-layer normalization) rather than Pre-LN as the default. The original BitNet paper found SubLN more stable than Pre-LN for binarized models and enables larger learning rates. Implement this as RMSNorm placed before each sub-layer (attention, FFN) within the residual stream.

Do not use hidden high-precision linear weight paths.

Avoid bias terms unless experiments prove they are necessary.

### 8.2 Activation Quantization Roadmap (Later Phases)

The BitNet a4.8 and BitNet v2 papers (2024–2025) have established a clear activation quantization progression:

1. 8-bit activations (INT8 per-token AbsMax) — baseline, minimal quality loss
2. 4-bit activations for attention inputs and FFN gate/up inputs (Gaussian-like distributions)
3. 8-bit with sparsification for intermediate states W_o and W_down (outlier-heavy distributions)
4. Full 4-bit activations via Hadamard pre-transformation (H-BitLinear) to smooth outlier distributions

This project should reach phase 1 before any work on activation quantization begins. Phases 2–4 are optional experiments documented in the experiment matrix.

---

## 9. Attention Architecture

Start with a decoder-only Transformer.

Recommended initial components:

- causal self-attention
- RoPE
- GQA where architecture size supports it
- SubLN (RMSNorm before each sub-layer)
- gated MLP (SwiGLU)

The attention architecture itself is not the research novelty. Keep it conventional enough that observed failures can be attributed to the binary training regime rather than an exotic architecture.

All learned matrix projections (Q, K, V, O, gate, up, down) should be binary in the target model.

---

## 10. Embeddings

Embeddings are explicitly included in the 1-bit target.

Do not leave the token embedding matrix in FP16/BF16 merely because it is convenient.

The embedding matrix should be:

```text
[vocab_size, hidden_size]
```

with both dimensions chosen to make the storage regular.

This is important because the project is specifically interested in end-to-end binary parameter storage, as demonstrated by Bonsai 8B (embeddings included in 1-bit representation).

The initial implementation may use higher precision temporarily inside the embedding lookup only if this is a technical necessity for an experiment, but the effective embedding table must remain binary and such deviations must be measured.

---

## 11. Tokenizer

Do not depend on a third-party tokenizer library.

Implement the tokenizer explicitly in Python using the standard library and NumPy where useful.

For the first practical implementation, use a byte-oriented scheme so that arbitrary input text has deterministic coverage. A byte-level BPE implementation is preferred once the proof-of-concept training pipeline is stable.

Choose a power-of-two vocabulary target:

```text
32768
65536
131072
```

Do not use 151,936 merely because Bonsai/Qwen uses it. Matching the teacher vocabulary exactly is not required for behavioral distillation because teacher and student can have separate tokenizers and the distillation process can operate over corresponding text responses.

If a direct teacher-logit distillation loss is desired, use the same tokenizer for that teacher/student experiment rather than silently assuming token identities correspond.

Document tokenizer version and hash in every experiment.

---

## 12. Bonsai Teacher Strategy

Bonsai should be used as a teacher only after the native student can learn basic language modeling.

Do not start by copying Bonsai hidden states into a blank network.

Use a staged approach.

### Phase A: Native Pretraining Without Teacher

Train the student from random initialization using next-token prediction combined with autoregressive distillation loss against a lightweight FP16 teacher (not necessarily Bonsai — any comparable model on the same or aligned tokenizer works here).

Purpose:

- verify binary optimization via STE + AD-loss
- verify convergence
- identify architectural instabilities
- establish a language-modeling baseline

### Phase B: Instruction Tuning

Use a clean instruction-response dataset and train the native binary model to predict assistant tokens.

Keep the same binary forward path.

### Phase C: Bonsai Behavioral Distillation

Feed the same prompt/corpus examples to both teacher (Bonsai) and student.

The initial distillation objective:

```text
L = lambda_ce * CE(student, ground_truth)
  + lambda_kd * KL(student_distribution, teacher_distribution)
```

Only introduce hidden-state matching after the logits/response distillation path is functioning.

### Teacher Inference

Cache Bonsai outputs where possible. Use Bonsai via GGUF or MLX format (both are available under Apache 2.0 for the 8B and 27B models). The student implementation itself must remain pure PyTorch/NumPy.

Both Bonsai 8B (1.15 GB) and Bonsai 27B (3.9 GB at 1-bit) are available as GGUF files suitable for local inference on reasonable hardware.

---

## 13. Tokenizer Issue for Distillation

Do not blindly compare teacher and student logits if their tokenizers differ.

Three safe strategies, in order of implementation simplicity:

1. **Strategy 1 (simplest):** Distill teacher-generated text into the student as ordinary supervised examples. No tokenizer alignment needed.
2. **Strategy 2:** Use the same tokenizer during the distillation experiment.
3. **Strategy 3:** Implement an explicit token-alignment / vocabulary projection mechanism.

Start with Strategy 1. Implement Strategy 2 as a controlled research experiment.

---

## 14. Instruction Tuning Data

The final target is an instruct model, not merely a base language model.

Use a mixture of:

- instruction-response examples
- conversational examples
- reasoning examples
- coding examples
- factual question answering
- tool-use formatted examples if the final interface needs tools

Every training example should reduce to a sequence with explicit roles:

```text
system
user
assistant
```

Loss masking should normally apply only to assistant tokens during supervised instruction tuning.

Keep the ability to run a pure language-modeling baseline on the same model.

---

## 15. `model.weights` File Format

Create a custom binary checkpoint format called `model.weights`.

Do not use PyTorch `.pt` or `.pth` files as the canonical model format.

### 15.1 Design Goals

The format must support:

- memory mapping
- deterministic offsets
- power-of-two alignment
- sequential layer reads
- direct tensor slicing
- packed 1-bit weights
- optional scale arrays
- shape metadata
- dtype metadata for non-binary tensors
- versioning
- integrity checks
- forward compatibility

### 15.2 Recommended Layout

```text
model.weights

[fixed header: magic bytes, version, model config]
[fixed-size tensor directory]
[optional metadata region]
[aligned tensor data]

  token_embedding
  layer_000.attn.q
  layer_000.attn.k
  layer_000.attn.v
  layer_000.attn.o
  layer_000.mlp.gate
  layer_000.mlp.up
  layer_000.mlp.down
  layer_000.norm_attn
  layer_000.norm_ffn
  layer_001....
  ...
  final_norm
  lm_head
```

Every major tensor should begin at an explicit alignment boundary.

Start with 4096-byte alignment (natural VM/page-sized boundary). Benchmark 64 KiB and 2 MiB alternatives before changing the default.

Do not assume that larger alignment automatically means faster I/O. Measure it.

### 15.3 Tensor Directory

Every tensor descriptor should include at least:

```text
name
rank
shape
storage type
bit width
file offset
byte length
alignment
optional scale offset
optional scale length
optional scale group size
dataset/model role
```

Use fixed-width integer fields wherever practical.

Avoid variable-length parsing in the hot loading path.

### 15.4 Binary Tensor Packing

For a binary tensor:

```text
bit_count = number_of_elements
byte_count = ceil(bit_count / 8)
```

The last byte must have deterministic unused-bit behavior.

Recommended rule: unused high bits are zeroed.

### 15.5 Scaled Binary Format

For the scaled experiment:

```text
[packed sign bits][FP16/BF16 AbsMean scale array]
```

Scale groups should be power-of-two sized. Default: 128 weights.

This matches the Bonsai 8B format (1 sign bit per weight, 1 FP16 scale per 128 weights = 1.125 bits/weight effective). The scale array must be independently addressable.

### 15.6 No JSON in the Hot Path

Human-readable JSON may be stored as a sidecar file for experiments, but the binary model format should use fixed binary metadata.

Do not force the runtime to parse JSON before accessing the first tensor.

---

## 16. File I/O and Memory Mapping

Implement a loader that can:

1. open `model.weights`
2. validate the header
3. map the file with `mmap`
4. resolve tensor descriptors
5. expose zero-copy views over packed data where possible
6. decode only the data required for a requested operation

Do not deserialize the entire model into Python lists.

Measure:

- file open time
- metadata parse time
- mmap time
- first-token latency
- per-layer weight access
- cold-cache vs warm-cache behavior
- sequential vs random tensor access
- peak resident memory

---

## 17. Kernel Strategy

Do not write custom CUDA/C++ kernels in the first implementation.

First validate correctness using explicit PyTorch tensor operations.

### 17.1 Binary Matrix Multiplication Reference

For binary vectors interpreted as `{-1, +1}`:

```text
XNOR(a, b)
       ↓
POPCOUNT
       ↓
2 * matches - N
       ↓
dot product
```

This identity comes from the fact that for bits a, b ∈ {0,1} mapped to {-1,+1}, the signed product is 1 if a==b and -1 otherwise, which is equivalent to XNOR followed by popcount.

**Important caveat:** Do not assume XNOR-popcount is automatically faster in PyTorch's eager mode. Benchmark it. PyTorch does not automatically lower to hardware popcount instructions. The speedup is realized only with custom kernels (like bitnet.cpp's I2_S, TL1, TL2 kernels) or bitwise PyTorch operations backed by AVX-512 VNNI or ARM NEON.

The reference implementation should be correct; performance optimization comes after correctness is established.

### 17.2 Lookup Table Approach

The bitnet.cpp TL1/TL2 kernels use lookup tables over grouped weight patterns to avoid per-weight computation. This is a viable later optimization. Document the reference algorithm clearly enough that this lookup-table transformation is obvious.

---

## 18. Training Implementation Organization

Recommended repository structure:

```text
nue/
├── README.md
├── RESEARCH.md
├── pyproject.toml
├── nue/
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── binary.py          # packing/unpacking, sign, absmean scale
│   ├── quantization.py    # binary_quantize, STE, decoupled-STE
│   ├── layers.py          # BitLinear (binary+STE), SubLN
│   ├── attention.py       # causal attention, RoPE, GQA
│   ├── mlp.py             # SwiGLU MLP
│   ├── transformer.py     # full decoder-only model
│   ├── losses.py          # NTP loss, AD-loss, KL distillation
│   ├── optim.py
│   ├── checkpoint.py      # training checkpoint (PyTorch state)
│   ├── weights_format.py  # model.weights spec + serialization
│   ├── mmap_loader.py     # mmap inference loader
│   ├── generation.py
│   └── evaluation.py
├── scripts/
│   ├── train.py
│   ├── train_distill.py
│   ├── tokenize.py
│   ├── pack_weights.py
│   ├── inspect_weights.py  # print every param: name/shape/dtype/role/binary
│   ├── convert_weights.py
│   ├── generate.py
│   └── benchmark.py
├── tests/
│   ├── test_binary.py
│   ├── test_quantization.py
│   ├── test_layers.py
│   ├── test_attention.py
│   ├── test_checkpoint.py
│   ├── test_mmap.py
│   └── test_training.py
└── experiments/
    ├── configs/
    ├── logs/
    ├── checkpoints/
    └── results/
```

Do not build a giant monolithic training script.

---

## 19. Testing Requirements

Every low-level component must have explicit tests.

### Binary Packing Tests

Verify:

- True/False logical values round-trip exactly
- packed bits round-trip exactly
- tensor shapes are preserved
- unused tail bits are deterministic
- file offsets are aligned
- malformed files are rejected

### Binary Arithmetic Tests

For small matrices compare:

```text
binary reference multiplication (XNOR-popcount)
vs
ordinary FP32 multiplication after mapping False→-1, True→+1
```

Results must match exactly.

Also verify: absmean scaling is computed correctly per group.

### Model Tests

Verify:

- causal masking (no future token leakage)
- RoPE
- GQA
- SubLN
- binary projections
- embedding lookup
- LM head
- loss masking (only assistant tokens on SFT data)
- generation

### Native 1-Bit Assertion

During every test and training mode, provide a debug assertion that verifies the effective forward weights satisfy:

```text
unique values ⊆ {False, True}
```

or, after numerical mapping:

```text
unique values ⊆ {-1, +1}
```

Enable globally with an environment variable.

### AD-Loss Tests

Verify that the autoregressive distillation loss computes the correct KL divergence, that lambda weighting works, and that the combined loss backpropagates to the binary layer's master parameters.

---

## 20. Experiment Matrix

Build a clear set of controlled experiments rather than immediately optimizing one configuration.

### Experiment A: FP Baseline

A tiny ordinary FP Transformer using the same architecture.

Purpose: establish that the dataset, tokenizer, optimizer, and architecture are capable of learning.

### Experiment B: Strict Binary + STE, Pure NTP

Binary weights, no scale, no teacher. Pure next-token prediction.

Purpose: establish the hardest case as a lower bound.

### Experiment C: Strict Binary + STE + AD-Loss

Binary weights, no scale, with autoregressive distillation loss against a compact FP teacher.

Purpose: determine how much distillation improves binary training stability. This is the **primary training path** given FBI-LLM results.

### Experiment D: Scaled Binary + STE + AD-Loss

Binary sign bits plus FP16 AbsMean scale per 128 weights.

Purpose: measure how much magnitude information (the Bonsai format) improves quality over pure binary.

### Experiment E: Decoupled STE

Replace vanilla STE with the Decoupled STE variant (separate forward/backward temperatures).

Purpose: test whether better gradient estimation improves convergence.

### Experiment F: Native Binary Instruction Tuning

Take the trained binary base model and perform instruction tuning.

Purpose: determine whether instruction following survives binary constraints.

### Experiment G: Bonsai-Generated Supervision

Generate instruction-response examples with Bonsai and train the native binary model on them.

Purpose: test whether a 1-bit teacher improves the student's instruction capability.

### Experiment H: Teacher Distribution Distillation

Where token alignment is available, add teacher KL loss using Bonsai's output distribution.

Purpose: compare hard-label training against softer teacher supervision.

### Experiment I: Activation Quantization (Later Phase)

Apply INT8 per-token absmax quantization to activations.

Purpose: measure the quality/efficiency tradeoff of also quantizing activations.

### Experiment J: H-BitLinear / Hadamard Activations

Apply the BitNet v2 H-BitLinear approach (Hadamard transformation before 4-bit activation quantization) for the W_o and W_down intermediate states.

Purpose: evaluate whether the activation distribution smoothing technique transfers to the strict binary weight setting.

---

## 21. Evaluation

Each checkpoint should be evaluated on at least:

- training loss
- validation loss
- perplexity
- generation quality (manual inspection)
- instruction following (AlpacaEval or equivalent)
- basic reasoning
- arithmetic
- coding
- factual QA

For a small proof-of-concept model, do not overinterpret benchmark scores. Focus on learning curves and relative comparisons.

Primary comparison chain:

```text
FP baseline
native pure binary (no teacher)
native pure binary + AD-loss
native scaled binary + AD-loss
native binary + instruction tuning
native binary + Bonsai supervision
```

Also record per checkpoint:

- logical parameter count
- packed file size
- total checkpoint size
- bits/parameter (effective)
- load latency
- generation latency
- tokens/second
- peak RAM
- peak VRAM
- training throughput
- training memory

**Critical benchmark note (2026):** Short-form benchmarks (MMLU, etc.) do not expose the failure mode of aggressive quantization. Hard benchmarks (AIME-class math, LiveCodeBench, multi-step agentic tasks) are where compression costs show up. Always include at least one hard reasoning benchmark, even at the 125M scale, to detect early whether binary training is degrading compositional ability.

---

## 22. Model Size Accounting

Always distinguish:

### Logical Parameter Count

The number of model parameters.

### Binary Weight Storage

For pure binary weights:

```text
bits = parameter_count
bytes = ceil(parameter_count / 8)
```

### Scaled Binary Storage

For group size G = 128:

```text
weight_bits = parameter_count
scale_count = ceil(parameter_count / 128)
scale_bytes = scale_count * 2  (FP16)
total_bytes = ceil(parameter_count / 8) + scale_bytes
effective_bpw = (parameter_count + scale_count * 16) / parameter_count ≈ 1.125
```

This is the Bonsai/BitNet format. It gives 1.125 bits/weight for group size 128.

### Real Model File Size

Include:

- packed weights
- scales
- norms
- metadata
- headers
- alignment padding

Do not market or report the idealized bit count as the complete model size.

---

## 23. Training-State Checkpoint vs Deployment Checkpoint

Maintain two clearly separated concepts.

### Training Checkpoint

May contain:

- auxiliary master parameters (STE latent weights)
- optimizer state
- scheduler state
- RNG state
- tokenizer information
- experiment configuration

### Deployment Checkpoint

`model.weights` must contain only what is necessary for inference.

The deployment file should never require the continuous auxiliary training parameters.

This preserves the property that the deployed model is genuinely stored as a packed binary-weight model.

---

## 24. Avoiding Accidental Floating-Point Escape Hatches

Explicitly audit every learnable tensor.

The model should not secretly retain FP16/BF16 versions of:

- embedding weights
- attention weights
- MLP weights
- output head weights

Add an inspection utility (`inspect_weights.py`) that prints every parameter with:

```text
name
shape
dtype
logical role
binary/non-binary status
trainable status
serialized status
absmean scale (if applicable)
```

This utility should be run automatically before training begins and before any checkpoint is written.

---

## 25. What OpenCode Should NOT Do

Do not:

- silently substitute an existing BitNet implementation from Hugging Face or elsewhere
- import Hugging Face Transformers to create the model
- use bitsandbytes for quantization
- call a quantization API without exposing the exact operation
- train an FP checkpoint and quantize it afterward
- serialize binary weights as one byte per boolean
- use Python lists as the persistent weight representation
- assume `bool` has one-bit memory semantics in Python
- add arbitrary dimensions when a power-of-two alternative is practical
- add hidden FP16/BF16 weight matrices for quality unless explicitly recorded as an ablation
- optimize for speed before correctness is established
- write custom CUDA kernels before the reference algorithm is tested
- treat Bonsai's proprietary compression/training procedure as known
- claim that reproducing Bonsai outputs means reconstructing Bonsai's architecture or weights
- skip AD-loss: the distillation objective is a first-class training concern for pure binary models, not an optional late experiment
- assume that short-form benchmark scores indicate the model is working well — always include a hard reasoning benchmark

---

## 26. Development Sequence

OpenCode should execute the work in this order.

### Step 1: Repository and Configuration

Create the repository structure and a central configuration system.

Every experiment must be reproducible from a configuration file.

### Step 2: Tokenizer

Implement and test the tokenizer independently.

### Step 3: Binary Representation

Implement packing, unpacking, binary logical tensors, AbsMean scaling, and exact tests.

### Step 4: Binary Linear Layer

Implement BitLinear: sign quantization in the forward pass, AbsMean scale (per group), STE backward pass.

Verify against ordinary FP arithmetic after the {-1,+1} mapping.

### Step 5: Optimization Mechanism

Implement STE as the first baseline.

Create a tiny toy problem where a binary layer must learn a known mapping.

Do not proceed until this demonstrably learns.

Then implement the Decoupled STE variant as experiment E.

### Step 6: AD-Loss

Implement the autoregressive distillation loss: a weighted sum of cross-entropy against ground truth and KL divergence against a teacher model's output distribution.

Verify it backpropagates through the binary layer correctly.

### Step 7: Tiny Transformer

Build the complete decoder-only model: SubLN, RoPE, GQA, SwiGLU, binary projections throughout.

Verify causal language modeling on a tiny corpus.

### Step 8: Native Pretraining

Train a small model from random initialization.

Confirm binary effective weights from step zero.

Use AD-loss with a compact FP teacher from this step forward.

### Step 9: Custom Checkpoint

Implement `model.weights` and mmap loading.

Verify that loading the deployment checkpoint reproduces inference from the training checkpoint to numerical tolerance.

### Step 10: Instruction Tuning

Train the binary model on instruction data.

### Step 11: Bonsai Teacher Pipeline

Add Bonsai-generated instruction data and later KL/distribution distillation.

### Step 12: Evaluation and Scaling

Run the experiment matrix.

Scale only after the tiny model is stable.

---

## 27. Resolved Design Decisions

### Q1. Binary or Ternary?

**Binary only** `{-1, +1}` for the primary model.

Ternary BitNet b1.58 is a reference and optional comparison experiment, not the target.

### Q2. Floating-Point Auxiliary Parameters?

**Yes, only as an optimizer mechanism.**

First baseline uses master parameter + STE. The forward path always uses binary weights.

### Q3. Binary Embeddings?

**Yes** for the final target. Bonsai 8B demonstrates this is viable.

### Q4. Binary LM Head?

**Yes** for the final target.

### Q5. Binary Activations?

**No for the first project phase.** Use FP16/BF16. Add activation quantization only as Experiment I/J.

### Q6. Power-of-Two Dimensions?

**Yes** wherever practical.

### Q7. File Alignment Power of Two?

**Yes.** Start at 4096 bytes.

### Q8. Custom Checkpoint Format?

**Yes.** `model.weights` is the canonical deployment format.

### Q9. PyTorch Checkpoint Files?

**Only for training/debug state**, never as the canonical deployment representation.

### Q10. Copy Bonsai Architecture?

**No.** Student starts randomly initialized. Architecture is independently designed.

### Q11. Bonsai Supervision During Pretraining?

**No.** Use a generic FP teacher for AD-loss during pretraining. Prove native binary language modeling first.

### Q12. How Bonsai Contributes?

**Phase 1:** Generate high-quality instruction-response supervision.
**Phase 2:** Logit/distribution distillation.
**Phase 3 (optional):** Hidden-state distillation.

### Q13. Share Bonsai's Tokenizer?

**No requirement.** Student tokenizer independently controlled with power-of-two vocabulary. Same-tokenizer distillation is a controlled experiment.

### Q14. Training Framework?

**PyTorch.**

### Q15. Inference Framework?

**PyTorch initially.** Python runtime until representation and algorithms are stable.

### Q16. One Byte Per Boolean?

**Absolutely not.** Pack 8 bits per byte.

### Q17. FP Scales in File?

**Optional depending on experiment.** Pure-binary experiments have no scales. Scaled-binary experiments use one FP16 AbsMean scale per group of 128.

### Q18. Clone Qwen3/Bonsai Architecture Exactly?

**No.** Use conventional decoder-only Transformer with power-of-two dimensions.

### Q19. First Target Size?

**~50M–125M parameters.**

### Q20. When to Attempt 8B?

Only after binary training algorithm, tokenizer, file format, and instruction tuning pipeline all work on smaller models.

### Q21. (Added) Should AD-Loss Be the Default Training Objective?

**Yes.** FBI-LLM results make it clear that pure NTP loss is significantly harder for binary training from scratch. AD-loss (combined NTP + KL distillation against any FP teacher) should be the default from Step 8 onward. Pure NTP (Experiment B) is retained as a lower-bound baseline only.

### Q22. (Added) AbsMean or AbsMax for Weight Scaling?

**AbsMean for per-group weight scales.** AbsMax for per-token activation quantization (when applicable). This matches the BitNet b1.58 empirical results and the production Bonsai format.

---

## 28. Success Criteria

Minimum success requires all of the following:

1. A model can start from random initialization.
2. The first forward pass uses binary weights.
3. The model can reduce training loss on a language modeling task.
4. The model can learn a meaningful validation distribution.
5. Instruction tuning improves instruction-following metrics.
6. The final deployment checkpoint contains packed binary weights.
7. The deployment checkpoint can be memory mapped and loaded without reconstructing an FP model checkpoint.
8. The deployed model reproduces the expected output behavior of the trained binary model.
9. The implementation does not rely on hidden third-party quantization/model libraries.
10. The repository contains tests proving the binary representation and checkpoint format are correct.

A stronger success criterion is that the Bonsai-guided version substantially improves instruction-following performance over the same binary model without Bonsai supervision.

A further criterion (2026): hard benchmark performance (AIME-class, LiveCodeBench) should not collapse relative to the FP baseline more than short-form benchmarks suggest. This is the failure mode seen with post-training quantization methods and should be tracked from the first training run.

---

## 29. Baseline Measurements Before Optimization

Before attempting low-level performance work, collect baseline numbers for:

- tokens/sec training
- tokens/sec inference
- peak VRAM
- peak RAM
- model file size
- model load time (cold cache and warm cache)
- first-token latency
- steady-state token latency
- validation loss
- instruction evaluation score
- hard reasoning benchmark score

The point of this project is both modeling and systems design. Accuracy and storage/I/O efficiency must be measured together.

---

## 30. Final Design Philosophy

The model should be viewed as a binary computational system rather than a conventional floating-point LLM with a quantizer attached.

The desired end state:

```text
random initialization
        ↓
true native binary forward weights
        ↓
native binary training (STE + AD-loss from a FP teacher)
        ↓
base language model
        ↓
native binary instruction tuning
        ↓
Bonsai-guided behavioral refinement (distribution distillation)
        ↓
packed model.weights (AbsMean scales per 128 weights)
        ↓
mmap / direct binary loading
        ↓
bit-oriented inference (XNOR-popcount path)
```

The core project question is not "how do we quantize an LLM to 1 bit?"

It is:

> "How do we design, train, store, and run an instruction-following Transformer whose fundamental learned parameters are binary from the first optimization step?"

OpenCode should treat this document as the specification. When an implementation choice is not explicitly specified, prefer the simplest implementation that preserves these principles and document the choice rather than introducing a large dependency.

---

## Appendix A: Key Empirical Numbers (2025–2026)

| Model | Bits/weight | Size (2B equiv) | Quality vs FP16 |
|---|---|---|---|
| Bonsai 8B (1-bit) | 1.125 | 1.15 GB | Comparable on standard benchmarks |
| Bonsai 27B (1-bit) | 1.125 | 3.9 GB | 89.5% of FP16 |
| Bonsai 27B (ternary) | ~1.7 | 5.9 GB | 94.6% of FP16 |
| BitNet b1.58 2B4T | 1.58 | 0.4 GB* | Matches FP16 LLaMA at 3B |
| FBI-LLM 7B | 1.0 | ~875 MB | Competitive with FP16 on perplexity |
| FP16 8B baseline | 16.0 | ~15 GB | 100% |

*Ternary packs to ~2 bits in practice with scales

bitnet.cpp inference speed (2B model, 8 CPU threads, x86):
- ARM CPUs: 1.37×–5.07× vs FP16
- x86 CPUs: 2.37×–6.17× vs FP16
- Energy reduction: 55%–82%

Bonsai 8B inference:
- M4 Pro Mac: 131 tok/s
- RTX 4090: 368 tok/s
- iPhone 17 Pro Max: ~44 tok/s

---

## Appendix B: STE Failure Modes and Mitigations

| Failure Mode | Symptom | Mitigation |
|---|---|---|
| Forward/backward mismatch | Loss plateaus early, weights oscillate | Gradient clipping; SubLN normalization |
| Depth scaling instability | Deeper models train worse than shallow | SubLN; layerwise progressive binarization |
| Large learning rate instability | NaN loss | SubLN; learning rate warmup; AbsMean normalization |
| Master weight drift from sign boundary | Binary weights all flip at once | Smaller learning rate; weight decay calibration |
| Teacher/student distribution gap | AD-loss KL never decreases | Ensure same tokenization; reduce lambda_kd initially |
