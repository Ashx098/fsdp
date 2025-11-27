# FSDP Training for GLM-4.5-Air MoE

Distributed FSDP training with LoRA fine-tuning for the GLM-4.5-Air Mixture-of-Experts model (106B parameters) on 8x Nvidia H200 GPUs.

## 🎯 Status: Production Ready ✅

- ✅ Stable training (87GB/141GB per GPU)
- ✅ CPU offload checkpointing (saves in ~3 min)
- ✅ Checkpoint resume verified
- ✅ Batch size 4 (4x throughput vs initial)

## Quick Start

### Training from Scratch

```bash
cd "/workspace/MoE Training/fsdp"
source /workspace/MoE\ Training/moe_env/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_TIMEOUT=7200 \
accelerate launch --config_file accelerate_config.yaml \
  train_fsdp_lora.py --config config.yaml
```

### Resume from Checkpoint

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_TIMEOUT=7200 \
accelerate launch --config_file accelerate_config.yaml \
  train_fsdp_lora.py --config config.yaml \
  --resume_from_checkpoint glm_output/checkpoint-10
```

## Key Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPUs | 8x H200 (141GB each) | 87GB used per GPU |
| Batch Size | 4 per GPU | 32 global batch |
| Sequence Length | 1024 tokens | Reduced from 4096 |
| LoRA Rank | 16 | 32M trainable params |
| FSDP Policy | SIZE_BASED_WRAP | 500M param threshold |
| Checkpoint Freq | Every 10 steps | ~3 min save time |

## Performance

- **Training Speed**: ~0.07 it/s (7 iterations/min)
- **Memory Usage**: 87GB/141GB per GPU (62% utilization)
- **Checkpoint Size**: 210GB (sharded) + 121MB (LoRA adapters)
- **Stability**: No OOM, no NCCL timeouts, no FSDP errors

## Files

- **`train_fsdp_lora.py`**: Main training script with CPU offload checkpointing
- **`config.yaml`**: Training hyperparameters and data settings
- **`accelerate_config.yaml`**: FSDP and distributed configuration
- **`DOCUMENTATION.md`**: Complete technical documentation ⭐
- **`GLM_TRAINING_GUIDE.md`**: Original training guide

## Features

### ✅ Working
- Stable distributed training across 8 GPUs
- CPU offload checkpoint saving (no OOM/timeout)
- Checkpoint resume with full state restoration
- Gradient checkpointing for memory efficiency
- Expert usage logging for MoE layers
- JSONL training logs

### ⚠️ Known Limitations
- Gradient accumulation > 1 causes FSDP state error (kept at 1)
- Evaluation disabled (needs synchronization fixes)
- Flash Attention 2 not installed (using SDPA fallback)
- Sequence length limited to 1024 (memory constraints)

## Architecture

**Model**: GLM-4.5-Air MoE
- Total Parameters: 106.9B
- Trainable (LoRA): 31.6M (0.03%)
- Experts per Layer: 96
- Top-K Routing: 2

**LoRA Configuration**:
- Rank: 16
- Alpha: 32
- Target Modules: `query_key_value`, `dense`
- Dropout: 0.05

## Critical Implementation Details

### CPU Offload Checkpointing ⭐

**Problem**: Standard checkpointing methods failed:
- `accelerator.get_state_dict()`: 10+ min timeout (gathering 212GB)
- `get_peft_state_dict()`: GPU OOM (tries to clone on GPU)

**Solution**: Dual checkpoint strategy
1. **Sharded state** via `accelerator.save_state()` (fast, parallel)
2. **LoRA adapters** via CPU offload (avoids GPU OOM)

```python
# Key code pattern
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state_dict = model.state_dict()  # Gathers to CPU RAM (1TB+)
    # Filter and save LoRA params...
```

**Why it works**: Uses abundant CPU RAM (1TB+) instead of limited GPU VRAM (141GB).

### FSDP Wrapping

**Critical**: Use `SIZE_BASED_WRAP` with `fsdp_min_num_params: 500000000`

- ❌ `TRANSFORMER_BASED_WRAP`: Causes severe OOM
- ✅ `SIZE_BASED_WRAP`: Balanced sharding across GPUs

### Gradient Flow Fix

Required for LoRA + gradient checkpointing:

```python
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()  # Critical!
model = model.to(torch.bfloat16)
```

## Troubleshooting

### OOM Errors
1. Check FSDP policy is `SIZE_BASED_WRAP`
2. Verify `block_size: 1024` in config.yaml
3. Confirm `batch_size: 4` or lower
4. Ensure `use_cache: false` in model config

### NCCL Timeout
1. Set `NCCL_TIMEOUT=7200` environment variable
2. Verify `fsdp_ddp_timeout: 7200` in accelerate_config.yaml
3. Check network connectivity between GPUs

### Checkpoint Hangs
1. Ensure using latest `train_fsdp_lora.py` with CPU offload
2. Verify `fsdp_state_dict_type: SHARDED_STATE_DICT`
3. Check `accelerator.wait_for_everyone()` calls present

## Documentation

📖 **For complete details**, see **[DOCUMENTATION.md](./DOCUMENTATION.md)** which includes:
- All problems encountered and solutions
- Detailed observations and findings
- Performance analysis
- Configuration explanations
- Best practices and lessons learned

## Requirements

```bash
pip install torch transformers accelerate peft datasets pyyaml
```

See `requirements.txt` for specific versions.

## Monitoring

```bash
# Watch training logs
tail -f training_output_new.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh glm_output/checkpoint-*
du -sh glm_output/checkpoint-*
```

## Using Trained Adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "glm_output/checkpoint-50")
tokenizer = AutoTokenizer.from_pretrained("glm_output/checkpoint-50")

# Inference
model.eval()
# ... use for generation
```

## Environment

- Python: 3.12.3
- PyTorch: 2.9.1+cu128
- Accelerate: 1.11.0
- CUDA: 12.8
- Hardware: 8x Nvidia H200 (141GB VRAM)

## License

Same as GLM-4.5-Air model license.

## Acknowledgments

- Implements CPU offload checkpointing strategy for FSDP+LoRA
- Based on HuggingFace Accelerate and PEFT libraries
- Optimized for Nvidia H200 GPUs

---

**Last Updated**: 2025-11-21  
**Status**: Production Ready ✅
