import os
import math
import json
import time
import shutil
import argparse
import yaml
import traceback
import logging
import gc
from pathlib import Path
from datetime import datetime
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from datetime import timedelta
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.distributed as dist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_error(error_log_path, error_type, error_message, step=None, epoch=None, traceback_str=None):
    """Log errors to error.jsonl file"""
    error_entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "error_type": error_type,
        "error_message": str(error_message),
        "step": step,
        "epoch": epoch,
        "traceback": traceback_str
    }
    try:
        with open(error_log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
    except Exception as e:
        print(f"Failed to log error: {e}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def log_expert_usage(model, outputs, step, accelerator, log_file_path=None):
    """Log MoE expert usage statistics"""
    if not (hasattr(outputs, "router_logits") and outputs.router_logits is not None):
        return
        
    unwrapped = accelerator.unwrap_model(model)
    config = unwrapped.config if hasattr(unwrapped, "config") else model.config
    
    num_experts = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", None)
    if num_experts is None:
        return

    num_layers = len(outputs.router_logits)
    layer_indices_to_log = [0, num_layers // 2]
    
    expert_usage_data = {}
    
    for layer_idx in layer_indices_to_log:
        if layer_idx >= len(outputs.router_logits):
            continue
            
        router_logit = outputs.router_logits[layer_idx]
        expert_choices = torch.argmax(router_logit, dim=-1)
        
        unique_experts, expert_counts = torch.unique(expert_choices.flatten(), return_counts=True)
        
        full_usage = torch.zeros(num_experts, dtype=torch.long, device=expert_choices.device)
        full_usage[unique_experts] = expert_counts
        
        total_usage = full_usage.sum()
        expert_usage_percent = full_usage.float() / total_usage * 100 if total_usage > 0 else full_usage.float()
        
        expert_usage_data[f"layer_{layer_idx}"] = {
            "expert_counts": {f"expert_{i}": int(count) for i, count in enumerate(full_usage)},
            "expert_percentages": {f"expert_{i}": round(float(percent), 2) for i, percent in enumerate(expert_usage_percent)},
            "total_tokens": int(total_usage)
        }
        
        if accelerator.is_main_process and (step % 100 == 0):
            accelerator.print(f"--- Step {step} | Layer {layer_idx} Expert Usage ---")
            used_experts = (full_usage > 0).sum().item()
            accelerator.print(f"  Active experts: {used_experts}/{num_experts}")
            accelerator.print("-" * 30)

    if log_file_path is not None and accelerator.is_main_process and (step % 100 == 0):
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "expert_usage": expert_usage_data
        }
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

def evaluate(model, eval_dataloader, accelerator):
    """Run evaluation with proper synchronization"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    accelerator.wait_for_everyone()
    
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            
            gathered_loss = accelerator.gather(loss.repeat(batch["input_ids"].size(0)))
            
            if accelerator.is_main_process:
                total_loss += gathered_loss.sum().item()
                total_samples += gathered_loss.size(0)
    
    if accelerator.is_main_process:
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    else:
        avg_loss = 0.0
        perplexity = 0.0
    
    model.train()
    accelerator.wait_for_everyone()
    
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}

def save_checkpoint(accelerator, model, tokenizer, output_dir, step, checkpoint_tracker, max_checkpoints, eval_loss=None):
    """
    Robust FSDP Checkpointing:
    1. Saves 'training_state' (optimizer + model) in SHARDED format for fast resuming.
    2. Saves 'adapter_model' (LoRA only) by gathering to CPU to avoid GPU OOM.
    """
    accelerator.wait_for_everyone()
    
    checkpoint_name = f"checkpoint-{step}"
    save_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- A. FAST RESUME CHECKPOINT (Sharded) ---
    # This saves the model and optimizer in chunks (shard_0, shard_1, etc.)
    # It is very fast and memory efficient. Use this to resume training.
    accelerator.print(f"Saving sharded training state for step {step}...")
    accelerator.save_state(save_dir)
    
    # --- B. EXPORT ADAPTERS (Consolidated) ---
    # This gathers ONLY the LoRA weights to CPU RAM on Rank 0 and saves standard adapter_model.bin
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)

    # Context manager: Tell FSDP to gather params to CPU, not GPU
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # This gathers the WHOLE model to CPU RAM.
        # H200 nodes usually have 1TB+ CPU RAM, so 212GB fits easily.
        cpu_state_dict = model.state_dict()
        
        if accelerator.is_main_process:
            # Filter: We only want LoRA params (trainable)
            lora_state_dict = {
                k: v for k, v in cpu_state_dict.items() 
                if "lora_" in k or "modules_to_save" in k
            }
            
            # Save the filtered dict using PEFT's format
            # We use unwrapped model to access the PEFT config
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                save_dir, 
                state_dict=lora_state_dict, 
                safe_serialization=True
            )
            
            # Save metadata
            metadata = {
                "step": step,
                "eval_loss": eval_loss,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Clean up CPU memory
            del cpu_state_dict
            del lora_state_dict
            gc.collect()
            
            print(f"✓ Saved LoRA adapters to {save_dir}")
            
            # Handle checkpoint rotation
            if eval_loss is not None:
                checkpoint_tracker[checkpoint_name] = eval_loss
                if len(checkpoint_tracker) > max_checkpoints:
                    sorted_checkpoints = sorted(checkpoint_tracker.items(), key=lambda x: x[1])
                    for ckpt_dir, _ in sorted_checkpoints[max_checkpoints:]:
                        ckpt_path = os.path.join(output_dir, ckpt_dir)
                        if os.path.exists(ckpt_path) and ckpt_dir != checkpoint_name:
                            shutil.rmtree(ckpt_path, ignore_errors=True)
                            print(f"Removed old checkpoint: {ckpt_dir}")
                    checkpoint_tracker.clear()
                    for ckpt_dir, val in sorted_checkpoints[:max_checkpoints]:
                        checkpoint_tracker[ckpt_dir] = val

    accelerator.wait_for_everyone()
    return checkpoint_name

def main():
    parser = argparse.ArgumentParser(description="GLM-4.5-Air FSDP+LoRA Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup directories
    output_dir = config['training']['output_dir']
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize Accelerator with extended timeout for large model synchronization
    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logs_dir)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision="bf16" if config['training']['bf16'] else "no",
        log_with="all",
        project_config=project_config,
        kwargs_handlers=[timeout_kwargs],
    )
    
    # Set seed
    set_seed(config['training']['seed'])
    
    if accelerator.is_main_process:
        accelerator.print(f"Output Directory: {output_dir}")
        accelerator.print(f"Logs Directory: {logs_dir}")
        accelerator.print(f"Number of processes: {accelerator.num_processes}")
        accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name_or_path'], 
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # But with LoRA, we need to load base model, then add adapters.
    accelerator.print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name_or_path'],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    
    # Disable cache for training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    # CRITICAL: Enable gradient checkpointing BEFORE LoRA
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    # CRITICAL: Enable input gradients for LoRA + Gradient Checkpointing
    model.enable_input_require_grads()
    
    # LoRA Config
    if accelerator.is_main_process:
        accelerator.print("✓ Gradient checkpointing enabled")
    
    # Add LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    
    # Ensure ALL parameters are in bf16
    for param in model.parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    # Dataset
    accelerator.print("Loading dataset...")
    dataset = load_dataset("json", data_files=config['data']['dataset_path'], split="train")
    
    # --- PACKING LOGIC START ---
    # Step 1: Tokenize without padding/truncation
    def tokenize_function(examples):
        # Just tokenize, don't pad/truncate yet
        return tokenizer(examples[config['data']['text_column']])

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=config['data']['preprocessing_num_workers'],
            desc="Tokenizing raw text"
        )

    # Step 2: Pack sequences by concatenation
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder at the end
        block_size = config['data']['block_size']
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # Create labels (same as input_ids)
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        tokenized_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=config['data']['preprocessing_num_workers'],
            desc="Packing sequences"
        )
    # --- PACKING LOGIC END ---
    
    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=config['data']['val_split'], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    accelerator.print(f"Train samples: {len(train_dataset)}")
    accelerator.print(f"Eval samples: {len(eval_dataset)}")
    
    def collate_fn(batch):
        # Simple stack, data is already uniform shape (packed to block_size)
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.ones_like(input_ids)  # All tokens are real now (no padding)
        labels = torch.stack([torch.tensor(item['labels']) for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn, 
        batch_size=config['training']['per_device_train_batch_size'],
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=collate_fn, 
        batch_size=config['training']['per_device_eval_batch_size'],
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer - only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Prepare with Accelerator FIRST (before calculating scheduler steps)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # CRITICAL: Calculate steps AFTER prepare() to get accurate sharded dataloader length
    # After prepare(), len(train_dataloader) returns the number of batches THIS rank will see
    # which already accounts for data sharding across GPUs
    num_epochs = config['training']['num_train_epochs']
    
    # Each step of train_dataloader processes one batch on this GPU
    # With gradient_accumulation_steps, we do N forward passes per optimizer step
    # So: optimizer_steps_per_epoch = len(train_dataloader) / gradient_accumulation_steps
    num_batches_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = num_batches_per_epoch // config['training']['gradient_accumulation_steps']
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    num_warmup_steps = int(0.03 * max_train_steps)
    
    # CRITICAL: The scheduler runs on each rank independently, so each rank steps it max_train_steps times
    # Do NOT prepare the scheduler - it doesn't need to be distributed and prepare() would divide steps by num_processes
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    accelerator.print(f"Batches per epoch (per rank): {num_batches_per_epoch}")
    accelerator.print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    accelerator.print(f"Optimizer steps per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"Total training steps (per rank): {max_train_steps}")
    accelerator.print(f"Warmup steps: {num_warmup_steps}")
    
    # Resume if needed
    start_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        try:
            start_step = int(args.resume_from_checkpoint.split("-")[-1])
            start_epoch = start_step // num_update_steps_per_epoch
            accelerator.print(f"Resuming from step {start_step}, epoch {start_epoch}")
        except:
            accelerator.print("Could not parse step from checkpoint name")
    
    # Clear memory before training
    gc.collect()
    torch.cuda.empty_cache()
    
    accelerator.print("=" * 50)
    accelerator.print("Starting training...")
    accelerator.print("=" * 50)
    
    global_step = start_step
    
    train_log_path = os.path.join(logs_dir, "train_log.jsonl")
    eval_log_path = os.path.join(logs_dir, "eval_log.jsonl")
    expert_log_path = os.path.join(logs_dir, "expert_usage.jsonl")
    error_log_path = os.path.join(logs_dir, "errors.jsonl")
    
    checkpoint_tracker = {}
    best_eval_loss = float('inf')
    
    accelerator.print("="*50)
    accelerator.print("Starting training...")
    accelerator.print("="*50)
    
    # Baseline evaluation at step 0
    if config['training'].get('eval_strategy') == "steps":
        accelerator.print("\n" + "="*50)
        accelerator.print("Running baseline evaluation (Step 0)...")
        accelerator.print("="*50)
        baseline_metrics = evaluate(model, eval_dataloader, accelerator)
        
        if accelerator.is_main_process:
            print(f"✓ Baseline Eval Loss: {baseline_metrics['eval_loss']:.4f} | Perplexity: {baseline_metrics['eval_perplexity']:.2f}")
            with open(eval_log_path, "a") as f:
                baseline_metrics['step'] = 0
                f.write(json.dumps(baseline_metrics) + "\n")
            best_eval_loss = baseline_metrics['eval_loss']
        
        accelerator.wait_for_everyone()
    
    start_time = time.time()
    log_loss = 0
    last_log_time = start_time
    steps_since_last_log = 0
    
    for epoch in range(start_epoch, int(num_epochs)):
        accelerator.print(f"\n{'='*50}")
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}")
        accelerator.print(f"{'='*50}")
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            # CRITICAL FIX: Wrap entire training step in try-except to catch OOM
            try:
                with accelerator.accumulate(model):
                    outputs = model(**batch, output_router_logits=True)
                    loss = outputs.loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        log_error(error_log_path, "NaNLoss", "NaN/Inf loss", global_step, epoch)
                        accelerator.print(f"⚠️  NaN/Inf loss at step {global_step}! Skipping...")
                        # Zero gradients and continue
                        optimizer.zero_grad()
                        continue
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        # Clip gradients
                        total_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        log_loss += loss.item()
                        
                        # Logging
                        if global_step % config['training']['logging_steps'] == 0:
                            avg_loss = log_loss / config['training']['logging_steps']
                            log_loss = 0
                            
                            # Calculate smooth speed (steps in last logging interval)
                            current_time = time.time()
                            time_since_last_log = current_time - last_log_time
                            smooth_speed = config['training']['logging_steps'] / time_since_last_log if time_since_last_log > 0 else 0
                            last_log_time = current_time
                            
                            # Calculate ETA
                            remaining_steps = max_train_steps - global_step
                            eta_seconds = remaining_steps / smooth_speed if smooth_speed > 0 else 0
                            eta_str = format_time(eta_seconds)
                            
                            lr = scheduler.get_last_lr()[0]
                            
                            if accelerator.is_main_process:
                                log_entry = {
                                    "step": global_step,
                                    "epoch": epoch + (step / len(train_dataloader)),
                                    "loss": avg_loss,
                                    "lr": lr,
                                    "steps_per_sec": smooth_speed,
                                    "grad_norm": float(total_norm) if total_norm is not None else 0.0,
                                    "eta_seconds": eta_seconds
                                }
                                
                                with open(train_log_path, "a") as f:
                                    f.write(json.dumps(log_entry) + "\n")
                                
                                print(f"Step {global_step}/{max_train_steps} | "
                                      f"Loss: {avg_loss:.4f} | "
                                      f"LR: {lr:.2e} | "
                                      f"Speed: {smooth_speed:.2f} it/s | "
                                      f"Grad: {float(total_norm):.2f} | "
                                      f"ETA: {eta_str}")
                        
                        # Expert Usage
                        if global_step % 100 == 0:
                            log_expert_usage(model, outputs, global_step, accelerator, expert_log_path)
                        
                        # Evaluation (separate from checkpointing)
                        just_evaluated = False
                        if config['training'].get('eval_strategy') == "steps" and global_step % config['training']['eval_steps'] == 0:
                            accelerator.print(f"\nRunning evaluation at step {global_step}...")
                            metrics = evaluate(model, eval_dataloader, accelerator)
                            eval_loss = metrics['eval_loss']
                            just_evaluated = True
                            
                            if accelerator.is_main_process:
                                print(f"✓ Eval Loss: {eval_loss:.4f} | Perplexity: {metrics['eval_perplexity']:.2f}")
                                with open(eval_log_path, "a") as f:
                                    metrics['step'] = global_step
                                    f.write(json.dumps(metrics) + "\n")
                                
                                if eval_loss < best_eval_loss:
                                    best_eval_loss = eval_loss
                                    print(f"🎉 New best eval loss: {best_eval_loss:.4f}")
                        
                        # Checkpointing
                        if global_step % config['training']['save_steps'] == 0:
                            # Use eval_loss from above if we just evaluated, otherwise None
                            checkpoint_eval_loss = best_eval_loss if (just_evaluated and best_eval_loss != float('inf')) else None
                            
                            # Save checkpoint
                            accelerator.print(f"Saving checkpoint at step {global_step}...")
                            save_checkpoint(
                                accelerator, model, tokenizer, output_dir, 
                                global_step, checkpoint_tracker, 
                                config['training']['save_total_limit'],
                                checkpoint_eval_loss
                            )
                            accelerator.print(f"✓ Checkpoint saved\n")
            
            except torch.cuda.OutOfMemoryError as e:
                # OOM handling
                error_msg = f"OOM at step {global_step}: {str(e)}"
                log_error(error_log_path, "OOM", error_msg, global_step, epoch, traceback.format_exc())
                accelerator.print(f"❌ {error_msg}")
                
                # Clear memory and reset gradients
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                
                # CRITICAL: Wait for all ranks to sync after OOM
                accelerator.wait_for_everyone()
                
                # Skip this batch and continue
                continue
                
            except Exception as e:
                # General error handling
                error_msg = f"Error at step {global_step}: {str(e)}"
                log_error(error_log_path, "TrainingError", error_msg, global_step, epoch, traceback.format_exc())
                accelerator.print(f"❌ {error_msg}")
                if accelerator.is_main_process:
                    traceback.print_exc()
                
                # Try to recover
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                continue
    
    accelerator.print("\n" + "="*50)
    accelerator.print("Training completed!")
    accelerator.print("="*50)
    
    # Final checkpoint
    save_checkpoint(
        accelerator, model, tokenizer, output_dir, 
        global_step, checkpoint_tracker, 
        config['training']['save_total_limit'],
        None
    )
    
    accelerator.end_training()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        raise
