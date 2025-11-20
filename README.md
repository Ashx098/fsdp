# FSDP + LoRA LLM Fine-tuning

This repository contains scripts for fine-tuning Large Language Models (LLMs) using Fully Sharded Data Parallel (FSDP) and Low-Rank Adaptation (LoRA).

## Setup

1.  **Environment**:
    ```bash
    source fsdp_env/bin/activate
    ```

2.  **Dependencies**:
    Dependencies are already installed in the virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configuration**:
    Edit `config.yaml` to set your model, dataset, and training parameters.

2.  **Training**:
    Run the training script:
    ```bash
    python train.py --config config.yaml
    ```

    For distributed training with Accelerate:
    ```bash
    accelerate launch train.py --config config.yaml
    ```

## Configuration (`config.yaml`)

-   **model**: Model name or path.
-   **data**: Dataset path and parameters.
-   **training**: Training hyperparameters (batch size, learning rate, epochs, etc.).
-   **lora**: LoRA configuration (rank, alpha, target modules).
-   **fsdp**: FSDP settings (wrapping policy, sharding strategy).

## Example

To run a quick test with GPT-2:
```bash
python train.py --config test_config.yaml
```
