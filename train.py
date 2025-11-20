import argparse
import yaml
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="FSDP + LoRA Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    config = load_config(args.config)

    if config['training'].get('report_to') == "none":
        os.environ["WANDB_DISABLED"] = "true"

    # Model loading
    print(f"Loading model: {config['model']['name_or_path']}")
    model_name = config['model']['name_or_path']
    
    # Determine torch dtype
    torch_dtype = torch.bfloat16 if config['training'].get('bf16', False) else torch.float16
    
    # Load model
    # Note: For FSDP, we usually load the model on CPU first or use device_map="auto" with care.
    # With HF Trainer + FSDP, it handles sharding.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        token=config['model'].get('use_auth_token', False),
        # device_map="auto" # FSDP doesn't always play nice with device_map="auto" initially
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        # Assuming json or text files for custom datasets
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
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=config['training']['bf16'],
        fp16=config['training']['fp16'],
        optim=config['training']['optim'],
        # FSDP specific arguments
        fsdp=config['fsdp'].get('fsdp', ""),
        fsdp_config=config['fsdp'].get('fsdp_config', None),
        remove_unused_columns=False, # Often needed for custom datasets
        ddp_find_unused_parameters=False, # Good practice for FSDP/DDP
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(config['training']['output_dir'])

if __name__ == "__main__":
    main()
