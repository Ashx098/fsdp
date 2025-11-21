import argparse
import yaml
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths in the CPT dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = config["data"]["dataset_path"]
    text_column = config["data"]["text_column"]
    num_proc = config["data"].get("preprocessing_num_workers", 1)

    if dataset_path == "wikitext":
        dataset = load_dataset(dataset_path, config["data"]["dataset_name"])
    else:
        dataset = load_dataset("json", data_files=dataset_path)

    def length_fn(examples):
        tokenized = tokenizer(
            examples[text_column],
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return {"length": [len(ids) for ids in tokenized["input_ids"]]}

    effective_num_proc = num_proc if num_proc and num_proc > 1 else None
    try:
        tokenized = dataset["train"].map(length_fn, batched=True, num_proc=effective_num_proc)
    except PermissionError:
        # Fallback without multiprocessing if worker locks are not permitted
        tokenized = dataset["train"].map(length_fn, batched=True, num_proc=None)
    lengths = tokenized["length"]

    arr = np.array(lengths)
    stats = {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    print("Sequence length stats (tokens):")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
