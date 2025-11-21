# Quick Reference Guide - FSDP Training

## Essential Commands

### Start Training
```bash
cd "/workspace/MoE Training/fsdp"
source /workspace/MoE\ Training/moe_env/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=7200 \
accelerate launch --config_file accelerate_config.yaml train_fsdp_lora.py --config config.yaml
```

### Resume Training
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=7200 \
accelerate launch --config_file accelerate_config.yaml train_fsdp_lora.py \
  --config config.yaml --resume_from_checkpoint glm_output/checkpoint-10
```

## Quick Checks

### Memory Usage
```bash
watch -n 1 nvidia-smi
```

### Training Progress
```bash
tail -f training_output_new.log
```

### Checkpoint Status
```bash
ls -lh glm_output/checkpoint-*
du -sh glm_output/checkpoint-*
```

## Key Settings

### Memory Optimization
- `block_size: 1024` (reduce if OOM)
- `per_device_train_batch_size: 4` (reduce if OOM)
- `gradient_checkpointing: true`

### FSDP Configuration
- `fsdp_auto_wrap_policy: SIZE_BASED_WRAP`
- `fsdp_min_num_params: 500000000`
- `fsdp_state_dict_type: SHARDED_STATE_DICT`

### Known Working Values
- Batch size: 4 → 87GB/GPU
- Batch size: 1 → ~65GB/GPU
- Block size 1024 → stable
- Block size 2048 → risky (125GB/GPU)

## Troubleshooting

### OOM on Startup
1. Reduce `per_device_train_batch_size` to 1
2. Reduce `block_size` to 512
3. Check FSDP policy is `SIZE_BASED_WRAP`

### NCCL Timeout
- Already set to 7200s (2 hours)
- Check network: `ping` other GPU nodes
- Monitor: `watch -n 1 "nvidia-smi | grep MiB"`

### Checkpoint Issues
- Use `train_fsdp_lora.py` (has CPU offload)
- Don't use old `train_glm.py`
- Verify `fsdp_state_dict_type: SHARDED_STATE_DICT`

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Speed | 0.05-0.10 it/s | 0.07 it/s ✅ |
| Memory | <100GB/GPU | 87GB/GPU ✅ |
| Checkpoint Time | <5 min | ~3 min ✅ |

## Important Files

- `train_fsdp_lora.py` - Main training script (USE THIS)
- `config.yaml` - Training settings
- `accelerate_config.yaml` - FSDP settings
- `README.md` - Quick start
- `DOCUMENTATION.md` - Complete details

## Critical Do's and Don'ts

### ✅ Do
- Use `SIZE_BASED_WRAP`
- Call `model.enable_input_require_grads()`
- Set `NCCL_TIMEOUT=7200`
- Test checkpoint resume early

### ❌ Don't
- Use `TRANSFORMER_BASED_WRAP`
- Set `gradient_accumulation_steps > 1`
- Skip `accelerator.wait_for_everyone()`
- Use `ddp_find_unused_parameters` with FSDP
