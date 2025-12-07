import torch
from utils.timer import Timer

def apply_kv_cache(model, tokenizer, prompt, max_new_tokens):
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Prefill phase
    with torch.no_grad():
        with Timer() as t:
            output = model(**inputs, use_cache=True)

    past_key_values = output.past_key_values

    # First next token (replaces argmax with top-k)
    next_token = torch.topk(output.logits[:, -1, :], k=1).indices

    generated_tokens = [next_token.item()]

    # Incremental decoding loop
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            output = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values
            )

        past_key_values = output.past_key_values

        # Use top-k instead of argmax
        next_token = torch.topk(output.logits[:, -1, :], k=1).indices
        generated_tokens.append(next_token.item())

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return decoded_text, t.elapsed
