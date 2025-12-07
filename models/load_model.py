from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("Model device:", next(model.parameters()).device)
    model.eval()
    return model, tokenizer

