# GLM-4.5-Air FSDP + LoRA Training Guide

This document provides a comprehensive guide to fine-tuning the **GLM-4.5-Air** Mixture-of-Experts (MoE) model using **Fully Sharded Data Parallel (FSDP)** and **Low-Rank Adaptation (LoRA)** on a cluster of **8x Nvidia H200 GPUs**.

## 1. System Overview

### Hardware Requirements
*   **GPUs**: 8x Nvidia H200 (141GB VRAM each).
*   **Total VRAM**: ~1.1TB.
*   **Interconnect**: NVLink recommended for efficient sharding.

### Software Stack
*   **Framework**: PyTorch 2.x + Hugging Face Transformers.
*   **Distributed**: Accelerate + FSDP.
*   **PEFT**: LoRA (Low-Rank Adaptation).
*   **Precision**: BFloat16 (BF16).

---

## 2. Model Architecture & Strategy

### The Model: GLM-4.5-Air
*   **Total Parameters**: 106 Billion.
*   **Active Parameters**: 12 Billion (per token).
*   **Architecture**: Mixture-of-Experts (MoE).
*   **Context Window**: 128k tokens.

### Training Strategy
To fit this 106B model into memory while maintaining high performance, we use:
1.  **FSDP (Full Shard)**: Shards model weights, optimizer states, and gradients across all 8 GPUs.
2.  **BF16 Precision**: Reduces memory usage by 50% compared to FP32, essential for H200s.
3.  **LoRA**: Fine-tunes only a small subset of parameters (Adapters), keeping the base model frozen.
4.  **Gradient Checkpointing**: Trades compute for memory by re-computing activations during the backward pass, allowing for longer sequence lengths (up to 16k-24k).

---

## 3. Configuration (`config.yaml`)

The configuration file controls all aspects of training.

### Model Section
```yaml
model:
  name_or_path: "zai-org/GLM-4.5-Air"
  trust_remote_code: True # Required for GLM custom architecture
```

### Training Parameters
*   **`bf16: True`**: **CRITICAL**. Do not change to False.
*   **`per_device_train_batch_size: 1`**: Keeps local memory usage low.
*   **`gradient_accumulation_steps: 4`**:
    *   Effective Batch Size = 8 (GPUs) * 1 (Local) * 4 (Accum) = **32**.
*   **`logging_steps: 20`**: Logs router statistics frequently.

### LoRA Configuration
*   **`target_modules`**: `["query_key_value", "dense", ...]`
    *   We target attention projections and MLP dense layers.
    *   **Note**: For MoE, targeting expert weights is possible but expensive.

### FSDP Configuration
```yaml
fsdp:
  fsdp_transformer_layer_cls_to_wrap: "GLMBlock"
  fsdp: "full_shard auto_wrap"
```
*   **`GLMBlock`**: Tells FSDP to wrap each transformer layer individually. This is crucial for memory management.

---

## 4. Custom Training Script (`train_glm.py`)

We use a custom script to handle specific requirements of MoE training.

### Key Features

#### 1. Router Logging
MoE models use a "router" to decide which experts process which tokens. Monitoring this is vital to ensure the model isn't collapsing (using only one expert).
*   **Metric**: `router/layerX_max_confidence`
*   **Interpretation**: High confidence means the router is sure. Low confidence might indicate instability.

#### 2. Perplexity Tracking
Standard loss is hard to interpret. We calculate **Perplexity** ($e^{loss}$) to measure how "surprised" the model is by the data.
*   **Goal**: Lower is better.

#### 3. Gradient Checkpointing
Enabled by default to maximize sequence length.

---

## 5. Running the Training

### Step 1: Activate Environment
```bash
source fsdp_env/bin/activate
```

### Step 2: Launch with Accelerate
Use `accelerate launch` to handle the distributed setup automatically.

```bash
accelerate launch train_glm.py --config config.yaml
```

**Note**: The script automatically detects all 8 GPUs.

---

## 6. Monitoring & Troubleshooting

### What to Watch (WandB)
1.  **Loss**: Should decrease over time.
2.  **Perplexity**: Should decrease.
3.  **Router Max Confidence**: Should remain stable (e.g., 0.6 - 0.9). If it drops to near 0.1 (random guessing), training might be unstable.

### Common Issues
*   **OOM (Out of Memory)**:
    *   **Fix**: Reduce `block_size` (sequence length) in `config.yaml`.
    *   **Fix**: Ensure `gradient_checkpointing` is True.
*   **Loss NaN/Inf**:
    *   **Fix**: Lower `learning_rate`.
    *   **Fix**: Check if `bf16` is enabled (FP16 can overflow).
