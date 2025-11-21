from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_path = "/workspace/Avinash/models/GLM-4.5-Air"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print("Instantiating model on meta device...")
try:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    print(f"Model class: {type(model).__name__}")
    
    # Find the decoder layers
    for name, module in model.named_modules():
        if "layers" in name and "." not in name.split("layers.")[-1]: # Top level layers list
            # usually model.layers or model.model.layers
            pass
        
        # Just print the first few modules to see structure
        if len(name.split(".")) < 4:
            print(f"{name}: {type(module).__name__}")
            
    # Specifically look for the block class
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(f"Layer class: {type(model.model.layers[0]).__name__}")
    elif hasattr(model, "layers"):
        print(f"Layer class: {type(model.layers[0]).__name__}")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"): # GLM style
        print(f"Layer class: {type(model.transformer.layers[0]).__name__}")
        
except Exception as e:
    print(f"Error: {e}")
