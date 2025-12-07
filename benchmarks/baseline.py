import torch
from utils.timer import Timer

def run_baseline(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with Timer() as t:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text, t.elapsed
