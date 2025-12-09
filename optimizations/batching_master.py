import time
import torch


# ----------------------------------------------------
# Helper: Measure execution time of a function
# ----------------------------------------------------
def measure(fn):
    start = time.time()
    out = fn()
    end = time.time()
    return out, (end - start)


# ----------------------------------------------------
# 1. SEQUENTIAL INFERENCE
# ----------------------------------------------------
def sequential_inference(model, tokenizer, prompts, max_new_tokens=80):
    """
    Runs each prompt one-by-one.
    Slowest method → demonstrates lack of batching.
    """
    results = []
    total_time = 0.0

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)

        def generate_fn():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids, duration = measure(generate_fn)

        results.append({
            "prompt": p,
            "output": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "time": duration
        })

        total_time += duration

    return {
        "method": "sequential",
        "total_time": total_time,
        "per_prompt": results
    }


# ----------------------------------------------------
# 2. STATIC BATCHING
# ----------------------------------------------------
def static_batch_inference(model, tokenizer, prompts, max_new_tokens=80):
    """
    All prompts are padded + batched together → one forward pass.
    Fastest for uniform shapes.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    def generate_fn():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids, duration = measure(generate_fn)

    outputs = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in output_ids
    ]

    return {
        "method": "static_batch",
        "batch_size": len(prompts),
        "total_time": duration,
        "prompts": prompts,
        "outputs": outputs
    }


# ----------------------------------------------------
# 3. PREFILL-MERGE BATCHING
# ----------------------------------------------------
import time
import torch

# ----------------------------------------------------
# Timing wrapper
# ----------------------------------------------------
def time_block(fn):
    def wrapper():
        start = time.time()
        out = fn()
        end = time.time()
        return out, end - start
    return wrapper


# ----------------------------------------------------
# 1. SEQUENTIAL INFERENCE
# ----------------------------------------------------
def sequential_inference(model, tokenizer, prompts, max_new_tokens=80):
    results = []
    total_time = 0.0

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)

        def generate_fn():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        (output_ids, duration) = time_block(generate_fn)()

        results.append({
            "prompt": p,
            "time": duration,
            "output": tokenizer.decode(output_ids[0], skip_special_tokens=True)
        })
        total_time += duration

    return {
        "method": "sequential",
        "total_time": total_time,
        "per_prompt": results
    }


# ----------------------------------------------------
# 2. STATIC BATCHING
# ----------------------------------------------------
def static_batch_inference(model, tokenizer, prompts, max_new_tokens=80):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    def generate_fn():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    (output_ids, duration) = time_block(generate_fn)()

    outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    return {
        "method": "static_batch",
        "batch_size": len(prompts),
        "total_time": duration,
        "prompts": prompts,
        "outputs": outputs
    }


# ----------------------------------------------------
# 3. PREFILL-MERGE BATCHING (FIXED)
# ----------------------------------------------------
def prefill_merge_inference(model, tokenizer, prompts, max_new_tokens=80):
    """
    Prefill-merge batching: Process each prompt's prefill phase separately,
    then merge for a single batched decode phase.
    
    This simulates scenarios where you want to compute KV caches independently
    before batching them together for generation.
    """
    
    # Track prefill time separately
    prefill_start = time.time()
    
    # Process each prompt individually to build KV caches
    all_input_ids = []
    all_attention_masks = []
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        
        # Validate input
        if inputs.input_ids.shape[1] == 0:
            continue
            
        all_input_ids.append(inputs.input_ids)
        all_attention_masks.append(inputs.attention_mask)
    
    if len(all_input_ids) == 0:
        return {
            "method": "prefill_merge",
            "error": "All prompts became empty after tokenization.",
            "outputs": [],
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "total_time": 0.0
        }
    
    # Find max length for padding
    max_length = max(ids.shape[1] for ids in all_input_ids)
    
    # Pad all inputs to same length
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pad_length = max_length - ids.shape[1]
        
        if pad_length > 0:
            # Pad on the left (for decoder-only models)
            padded_ids = torch.cat([
                torch.full((1, pad_length), tokenizer.pad_token_id, device=ids.device),
                ids
            ], dim=1)
            padded_mask = torch.cat([
                torch.zeros((1, pad_length), device=mask.device, dtype=mask.dtype),
                mask
            ], dim=1)
        else:
            padded_ids = ids
            padded_mask = mask
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Stack into batch
    batched_input_ids = torch.cat(padded_input_ids, dim=0)
    batched_attention_mask = torch.cat(padded_attention_masks, dim=0)
    
    prefill_time = time.time() - prefill_start
    
    # Now do batched generation (the "merge" phase)
    def decode_fn():
        with torch.no_grad():
            return model.generate(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True
            )
    
    (output_ids, decode_time) = time_block(decode_fn)()
    
    # Decode outputs
    decoded_outputs = [
        tokenizer.decode(seq, skip_special_tokens=True)
        for seq in output_ids
    ]
    
    return {
        "method": "prefill_merge",
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": prefill_time + decode_time,
        "outputs": decoded_outputs
    }



# ----------------------------------------------------
# 4. DYNAMIC BATCHING
# ----------------------------------------------------
def dynamic_batch_inference(
    model,
    tokenizer,
    prompts,
    max_new_tokens=80,
    batch_size=2
):
    """
    Groups prompts into runtime batches of size N.
    Demonstrates real production batching like vLLM / server batching.
    """
    all_batches = []
    current = []

    for idx, p in enumerate(prompts):
        current.append(p)
        if len(current) == batch_size or idx == len(prompts) - 1:
            all_batches.append(current)
            current = []

    all_outputs = []
    total_time = 0.0

    for batch in all_batches:

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        def generate_fn():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids, duration = measure(generate_fn)

        decoded = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]

        all_outputs.extend(decoded)
        total_time += duration

    return {
        "method": "dynamic_batch",
        "batch_size": batch_size,
        "num_batches": len(all_batches),
        "total_time": total_time,
        "outputs": all_outputs,
        "batches": all_batches
    }
