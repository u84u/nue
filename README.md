# nue

Native 1-bit Transformer. Weights are binary `{-1, +1}` from the first training step — no post-training quantization.

## What's here

A from-scratch decoder-only Transformer where every weight-bearing layer (embeddings, attention, MLP, LM head) uses sign-quantized binary weights with per-group AbsMean scaling. Trained with STE (Straight-Through Estimator) on CPU.

Current proof-of-concept: **1.25M parameters**, trained on Shakespeare.

```
nue/
├── tokenizer.py          # BPE tokenizer (no HF dependency)
├── binary.py             # sign quantize, AbsMean scaling, bit packing
├── layers.py             # BitLinear, BinaryEmbedding, SubLN
├── transformer.py        # decoder-only model, KV cache, RoPE, GQA
├── quantization.py       # STE, Decoupled STE
├── losses.py             # NTP + autoregressive distillation
├── weights_format.py     # custom .weights binary format
├── mmap_loader.py        # zero-copy mmap inference loader
├── assertions.py         # verify weights stay binary
└── data.py               # instruction data loading
```

## Quick start

```bash
# train
PYTHONPATH=. python implementations/1M/train.py

# resume training
PYTHONPATH=. python implementations/1M/train.py --resume

# chat with the model
PYTHONPATH=. python run.py implementations/1M/generated
```

REPL commands: `/temp 0.8`, `/topk 50`, `/tokens 512`, `/quit`

## Architecture

Power-of-two everything: vocab=1024, hidden=128, heads=4, kv_heads=2, head_dim=32, mlp=512, layers=4.

- Binary embeddings via STE
- SubLN (RMSNorm before each sub-layer)
- RoPE positional encoding
- GQA attention with fused QKV
- SwiGLU MLP with fused gate+up projection
- KV cache for autoregressive generation
- `torch.compile` + `inference_mode` for ~2x inference speedup

## Training

Loss is next-token cross-entropy. The `absmean_scale` function computes per-group AbsMean scaling (group_size=128) for the binary weights. The model uses Adam optimizer with cosine-warmup LR schedule.

Checkpoints save to `implementations/1M/generated/`. `rm -rf implementations/1M/generated` to start fresh.

## Binary weights

Every forward pass: `w_effective = AbsMean(sign(w_master))`. The STE allows gradients to flow to the master float32 parameters while keeping effective weights binary.

Custom `.weights` format stores packed bits with per-group FP16 scales (1.125 bits/weight effective). Memory-mapped loading for inference.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

26 tests covering binary packing, quantization, tokenizer, layers, checkpoint roundtrip, and transformer forward pass.

## Research context

This is a research implementation exploring whether native binary training from scratch can produce useful language models. References: BitNet, BitNet b1.58, FBI-LLM, Bonsai 8B. Full spec in [SPEC.md](SPEC.md).
