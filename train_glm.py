import argparse
import yaml
import os
import torch
import math
import json
import logging
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict
from datasets import load_dataset
from typing import Optional
import wandb

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class GLMTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save only LoRA adapters (PEFT) + tokenizer. Under FSDP, gather full params on rank0.
        """
        output_dir = output_dir or self.args.output_dir
        if not self.is_world_process_zero():
            return
        os.makedirs(output_dir, exist_ok=True)

        # Unwrap FSDP/Accelerate to access the underlying PEFT model
        unwrapped = getattr(self.model, "module", self.model)

        # Inspect adapter state dict to catch empty saves early (e.g., target_modules mismatch)
        try:
            adapter_sd = get_peft_model_state_dict(unwrapped)
            if len(adapter_sd) == 0:
                print("[rank0] Warning: Empty PEFT adapter state dict. Verify lora.target_modules match model.")
        except Exception as e:
            print(f"[rank0] Could not inspect PEFT state dict: {e}")

        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized() and isinstance(self.model, FSDP):
                # Gather full parameters onto CPU on rank0 so that save_pretrained sees consolidated weights
                from torch.distributed.fsdp import summon_full_params
                with summon_full_params(self.model, writeback=False, offload_to_cpu=True):
                    unwrapped.save_pretrained(output_dir, safe_serialization=True)
            else:
                # Non-FSDP or non-distributed path
                unwrapped.save_pretrained(output_dir, safe_serialization=True)
        except Exception as e:
            print(f"[rank0] save_pretrained failed on unwrapped model: {e}. Attempting direct model.save_pretrained.")
            try:
                self.model.save_pretrained(output_dir, safe_serialization=True)
            except Exception as e2:
                print(f"[rank0] Fallback model.save_pretrained also failed: {e2}")

        # Save tokenizer if available
        if getattr(self, "tokenizer", None) is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception as e:
                print(f"[rank0] tokenizer.save_pretrained failed: {e}")

        # Save trainer state on main process only
        try:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        except Exception as e:
            print(f"[rank0] Saving trainer_state.json failed: {e}")

        # Helpful listing of saved files and sanity check for adapter artifacts
        try:
            saved = os.listdir(output_dir)
            print(f"[rank0] Saved to {output_dir}: {saved}")
            missing = []
            for fname in ("adapter_model.safetensors", "adapter_config.json"):
                if fname not in saved:
                    missing.append(fname)
            if missing:
                print(f"[rank0] WARNING: Missing expected adapter files: {missing}")
        except Exception as e:
            print(f"[rank0] Listing saved files failed: {e}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Override compute_loss to access router logits if available.
        """
        outputs = model(**inputs)
        
        # Log router stats if available and we are in a logging step
        # Note: This is a simplification. Accessing 'router_logits' depends on the exact model output structure.
        # GLM-4.5-Air might return them in auxiliary_logits or similar.
        # We'll check for 'router_logits' or 'aux_loss'.
        
        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
            if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                # router_logits is usually a tuple of tensors (one per layer)
                # Shape: [batch_size, seq_len, num_experts]
                
                # Log max logit (confidence) and mean logit
                # We'll just take the first layer for brevity or average across layers
                try:
                    # Example: Log stats for the first MoE layer
                    first_layer_logits = outputs.router_logits[0] 
                    max_logits = first_layer_logits.max(dim=-1).values.mean().item()
                    mean_logits = first_layer_logits.mean().item()
                    
                    wandb.log({
                        "router/layer0_max_confidence": max_logits,
                        "router/layer0_mean_logit": mean_logits,
                        "train/global_step": self.state.global_step
                    })
                except Exception as e:
                    print(f"Failed to log router logits: {e}")

        loss = outputs.loss
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            loss = loss + outputs.aux_loss
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # This function is called on CPU after gathering, might be heavy for 106B model.
    # For perplexity, we usually just use the eval_loss from the trainer state.
    # But if we want to compute it explicitly here:
    # We can't easily compute perplexity here without the loss.
    # Standard Trainer logs eval_loss, so we can compute perplexity from that in the callback or just post-process.
    return {}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original logits are too large to gather on CPU (vocab size ~150k * seq len).
    We return just the loss or a subset if needed.
    For now, we rely on the Trainer's built-in loss logging for Perplexity calculation.
    """
    return logits


class JSONLLogger(TrainerCallback):
    """
    Write train/eval logs to jsonl files on the main process only.
    """

    def __init__(self, output_dir):
        self.train_path = os.path.join(output_dir, "train_log.jsonl")
        self.eval_path = os.path.join(output_dir, "eval_log.jsonl")
        os.makedirs(output_dir, exist_ok=True)

    def _write(self, path, payload):
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (getattr(args, "process_index", 0) != 0) or not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        self._write(self.train_path, record)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if (getattr(args, "process_index", 0) != 0) or not metrics:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(metrics)
        self._write(self.eval_path, record)

def main():
    parser = argparse.ArgumentParser(description="GLM-4.5-Air FSDP + LoRA Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    config = load_config(args.config)

    if config['training'].get('report_to') == "none":
        os.environ["WANDB_DISABLED"] = "true"

    # Local logging (console + file)
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    log_file = os.path.join(config['training']['output_dir'], "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    logger = logging.getLogger(__name__)

    # Model loading
    logger.info(f"Loading model: {config['model']['name_or_path']}")
    model_name = config['model']['name_or_path']
    
    torch_dtype = torch.bfloat16 if config['training'].get('bf16', False) else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        use_auth_token=config['model'].get('use_auth_token', False),
        # device_map="auto" # FSDP handles this
    )
    
    # Enable gradient checkpointing
    if config['training'].get('gradient_checkpointing', False):
        # use_reentrant=False is often safer with FSDP
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Force eager attention to avoid SDPA mask broadcast issues during recompute
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # Disable KV cache during training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Explicitly route PyTorch to math SDPA and disable flash/mem-efficient SDPA,
    # which can exhibit mask broadcasting issues during recompute.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = config['data']['block_size']

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules']
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # Ensure all parameters are a single dtype to satisfy FSDP flattening
    model = model.to(torch_dtype)

    # Dataset
    logger.info(f"Loading dataset: {config['data']['dataset_path']}")
    if config['data']['dataset_path'] == "wikitext":
         dataset = load_dataset(config['data']['dataset_path'], config['data']['dataset_name'])
    else:
        dataset = load_dataset("json", data_files=config['data']['dataset_path'])
    logger.info(f"Raw dataset sizes: {[f'{k}={len(v)}' for k,v in dataset.items()]}")
    # Train/validation split if not provided
    if "validation" not in dataset and config['data'].get('val_split', 0) > 0:
        split = dataset["train"].train_test_split(
            test_size=config['data']['val_split'],
            seed=config['training'].get('seed', 42)
        )
        split["validation"] = split.pop("test")
        dataset = split
        logger.info(f"Created validation split with test_size={config['data']['val_split']}")

    def tokenize_function(examples):
        return tokenizer(
            examples[config['data']['text_column']],
            padding=False,  # dynamic padding in collator to reduce memory
            truncation=True,
            max_length=config['data']['block_size'],
            return_attention_mask=True,
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config['data']['preprocessing_num_workers'],
        remove_columns=dataset["train"].column_names
    )
    logger.info("Tokenization complete.")
    # Sequence packing to fill block_size
    def pack_sequences(examples):
        packed_input_ids = []
        packed_attention_mask = []
        max_length = config['data']['block_size']
        current_ids = []
        total = 0
        for ids in examples["input_ids"]:
            length = len(ids)
            if total + length > max_length:
                if current_ids:
                    packed_input_ids.append(current_ids)
                    packed_attention_mask.append([1] * len(current_ids))
                current_ids = []
                total = 0
            current_ids += ids
            total += length
        if current_ids:
            packed_input_ids.append(current_ids)
            packed_attention_mask.append([1] * len(current_ids))
        return {"input_ids": packed_input_ids, "attention_mask": packed_attention_mask}

    tokenized_datasets = tokenized_datasets.map(
        pack_sequences,
        batched=True,
        num_proc=config['data']['preprocessing_num_workers'],
        remove_columns=tokenized_datasets["train"].column_names
    )
    train_len = len(tokenized_datasets["train"])
    val_len = len(tokenized_datasets["validation"]) if "validation" in tokenized_datasets else 0
    logger.info(f"Packed dataset sizes: train={train_len}, val={val_len}, block_size={config['data']['block_size']}, "
                f"per_device_train_batch_size={config['training']['per_device_train_batch_size']}, "
                f"grad_accum_steps={config['training']['gradient_accumulation_steps']}")
    has_validation = "validation" in tokenized_datasets

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training'].get('per_device_eval_batch_size', config['training']['per_device_train_batch_size']),
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['num_train_epochs'],
        weight_decay=config['training']['weight_decay'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        save_on_each_node=config['training'].get('save_on_each_node', True),
        bf16=config['training']['bf16'],
        fp16=config['training']['fp16'],
        optim=config['training']['optim'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        report_to=config['training']['report_to'],
        seed=config['training'].get('seed', 42),
        # FSDP specific arguments
        fsdp=config['fsdp'].get('fsdp', ""),
        fsdp_config=config['fsdp'].get('fsdp_config', None),
        remove_unused_columns=False, 
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        ddp_timeout=config['fsdp'].get('ddp_timeout', 3600),
        eval_accumulation_steps=config['training'].get('eval_accumulation_steps', 1),
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        dataloader_drop_last=config['training'].get('dataloader_drop_last', True),
    )
    total_steps = (len(tokenized_datasets["train"]) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * int(training_args.num_train_epochs)
    logger.info(f"Estimated total training steps: {total_steps}")

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
        tokenizer=tokenizer,
    )

    # Add Perplexity Logging Callback
    class PerplexityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            eval_loss = metrics.get("eval_loss")
            if eval_loss:
                perplexity = math.exp(eval_loss)
                print(f"Perplexity: {perplexity}")
                if args.report_to and "wandb" in args.report_to:
                    wandb.log({"perplexity": perplexity, "global_step": state.global_step})

    class PeftSaveCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
        def on_save(self, args, state, control, **kwargs):
            # Save adapters into the checkpoint dir on rank0 only
            if getattr(args, "process_index", 0) != 0:
                return
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            try:
                self.trainer.save_model(ckpt_dir)
            except Exception as e:
                print(f"PeftSaveCallback save_model failed: {e}")

    trainer.add_callback(JSONLLogger(training_args.output_dir))
    trainer.add_callback(PerplexityCallback())
    trainer.add_callback(PeftSaveCallback(trainer))

    # Optional: baseline evaluation before training (guarded by config to avoid NCCL timeouts)
    run_baseline_eval = config['training'].get('run_baseline_eval', False)
    if run_baseline_eval and "validation" in tokenized_datasets:
        eval_log_path = os.path.join(training_args.output_dir, "eval_log.jsonl")
        baseline_done = os.path.exists(eval_log_path) and os.path.getsize(eval_log_path) > 0
        if not baseline_done:
            if trainer.args.should_log and trainer.is_world_process_zero():
                print("Running baseline evaluation before training...")
            try:
                base_metrics = trainer.evaluate()
                if trainer.is_world_process_zero() and base_metrics:
                    print(f"Baseline eval metrics: {base_metrics}")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if trainer.is_world_process_zero():
                        print("Baseline eval OOM; skipping baseline evaluation. Training will proceed.")
                else:
                    raise

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(config['training']['output_dir'])

if __name__ == "__main__":
    main()
