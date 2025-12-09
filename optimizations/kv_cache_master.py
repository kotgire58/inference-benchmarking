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
    # PREFILL
    # ---------------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        with Timer() as prefill_timer:
            prefill_output = model(**inputs, use_cache=True)

    past = prefill_output.past_key_values  # DynamicCache

    next_token = torch.topk(prefill_output.logits[:, -1, :], k=1).indices
    generated_tokens = [next_token.item()]

    # ---------------------------------
    # Helper utilities
    # ---------------------------------
    def truncate(cache_tuples, size):
        return [
            (k[:, :, -size:], v[:, :, -size:]) for (k, v) in cache_tuples
        ]

    def enlarge(cache_tuples, factor):
        return [
            (
                torch.cat([k, torch.zeros_like(k).repeat(1, 1, factor, 1)], dim=2),
                torch.cat([v, torch.zeros_like(v).repeat(1, 1, factor, 1)], dim=2),
            )
            for (k, v) in cache_tuples
        ]

    # ---------------------------------
    # DECODE LOOP
    # ---------------------------------
    with Timer() as decode_timer:
        for _ in range(max_new_tokens - 1):

            if cache_mode == "optimal":
                past_used = past  # no change

            else:
                # Convert DynamicCache → legacy list-of-tuples
                legacy_cache = past.to_legacy_cache()

                if cache_mode == "small":
                    modified = truncate(legacy_cache, size=16)

                elif cache_mode == "large":
                    modified = enlarge(legacy_cache, factor=2)

                elif cache_mode == "custom":
                    modified = truncate(legacy_cache, size=custom_cache_size)

                # Convert modified tuples back → DynamicCache
                past_used = past.from_legacy_cache(modified)

            # Decode next token
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    past_key_values=past_used,
                    use_cache=True
                )

            past = out.past_key_values
            next_token = torch.topk(out.logits[:, -1, :], k=1).indices
            generated_tokens.append(next_token.item())

    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "output": decoded_output,
        "prefill_time": prefill_timer.elapsed,
        "decode_time": decode_timer.elapsed,
        "total_time": prefill_timer.elapsed + decode_timer.elapsed,
        "kv_mode": cache_mode,
        "cache_size": custom_cache_size
    }
