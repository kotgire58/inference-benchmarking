import torch
from utils.timer import Timer

def generate_with_kv_cache(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    cache_mode="optimal",
    custom_cache_size=None
):
    # ---------------------------------
    # ENCODE PROMPT (PREFILL)
    # ---------------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        with Timer() as prefill_timer:
            prefill_output = model(**inputs, use_cache=True)

    past = prefill_output.past_key_values

    # First next token
    next_token = torch.topk(prefill_output.logits[:, -1, :], k=1).indices
    generated_tokens = [next_token.item()]

    # ---------------------------------
    # CUSTOM CACHE TRANSFORM HELPERS
    # ---------------------------------
    def truncate_cache(past, size):
        return tuple(
            (
                k[:, :, -size:],   # keep last <size> keys
                v[:, :, -size:]    # keep last <size> values
            )
            for (k, v) in past
        )

    def enlarge_cache(past, factor):
        return tuple(
            (
                torch.cat([k, torch.zeros_like(k).repeat(1, 1, factor, 1)], dim=2),
                torch.cat([v, torch.zeros_like(v).repeat(1, 1, factor, 1)], dim=2),
            )
            for (k, v) in past
        )

    # ---------------------------------
    # INCREMENTAL DECODE LOOP
    # ---------------------------------
    with Timer() as decode_timer:
        for _ in range(max_new_tokens - 1):

            # APPLY KV-CACHE RULES
            if cache_mode == "small":
                past_used = truncate_cache(past, size=16)

            elif cache_mode == "large":
                past_used = enlarge_cache(past, factor=16)

            elif cache_mode == "custom":
                past_used = truncate_cache(past, size=custom_cache_size)

            else:
                past_used = past  # optimal

            # DECODE ONE TOKEN
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=past_used
                )

            past = out.past_key_values

            next_token = torch.topk(out.logits[:, -1, :], k=1).indices
            generated_tokens.append(next_token.item())

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "output": decoded,
        "prefill_time": prefill_timer.elapsed,
        "decode_time": decode_timer.elapsed,
        "total_time": prefill_timer.elapsed + decode_timer.elapsed,
        "kv_mode": cache_mode,
        "cache_size": custom_cache_size
    }
