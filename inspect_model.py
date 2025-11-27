from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_path = "/workspace/Avinash/models/GLM-4.5-Air"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"Model Config: {config}")

# We don't need to load the whole model to see the class structure if we trust the code, 
# but let's load on meta device if possible or just inspect the code file.
# Since trust_remote_code=True, the code is in the model folder.

import os
print(f"Model files: {os.listdir(model_path)}")
