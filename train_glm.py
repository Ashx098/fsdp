import argparse
import yaml
import os
import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class GLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to access router logits if available.
        """
        # Ensure router logits are outputted
        if "output_router_logits" not in inputs:
            inputs["output_router_logits"] = True
            
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

def main():
    parser = argparse.ArgumentParser(description="GLM-4.5-Air FSDP + LoRA Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    config = load_config(args.config)

    if config['training'].get('report_to') == "none":
        os.environ["WANDB_DISABLED"] = "true"

    # Model loading
    print(f"Loading model: {config['model']['name_or_path']}")
    model_name = config['model']['name_or_path']
    
    torch_dtype = torch.bfloat16 if config['training'].get('bf16', False) else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        token=config['model'].get('use_auth_token', True),
        # device_map="auto" # FSDP handles this
    )
    
    # Enable gradient checkpointing
    if config['training'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Dataset
    print(f"Loading dataset: {config['data']['dataset_path']}")
    if config['data']['dataset_path'] == "wikitext":
         dataset = load_dataset(config['data']['dataset_path'], config['data']['dataset_name'])
    else:
        dataset = load_dataset("json", data_files=config['data']['dataset_path'])

    def tokenize_function(examples):
        return tokenizer(examples[config['data']['text_column']], padding="max_length", truncation=True, max_length=config['data']['block_size'])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config['data']['preprocessing_num_workers'],
        remove_columns=dataset["train"].column_names
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['num_train_epochs'],
        weight_decay=config['training']['weight_decay'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps" if config['training']['eval_steps'] > 0 else "no",
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=config['training']['bf16'],
        fp16=config['training']['fp16'],
        optim=config['training']['optim'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        report_to=config['training']['report_to'],
        # FSDP specific arguments
        fsdp=config['fsdp'].get('fsdp', ""),
        fsdp_config=config['fsdp'].get('fsdp_config', None),
        remove_unused_columns=False, 
        ddp_find_unused_parameters=False,
    )

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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

    from transformers import TrainerCallback
    trainer.add_callback(PerplexityCallback())

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(config['training']['output_dir'])

if __name__ == "__main__":
    main()
