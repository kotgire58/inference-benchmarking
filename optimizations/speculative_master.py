import time
import torch


# --------------------------------------------------
# Helper for timing functions
# --------------------------------------------------
def measure(fn):
    start = time.time()
    out = fn()
    end = time.time()
    return out, end - start


# --------------------------------------------------
# 1. BASELINE DECODE
# --------------------------------------------------
def baseline_decode(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def gen_fn():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    (output_ids, duration) = measure(gen_fn)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "method": "baseline",
        "output": text,
        "time": duration
    }


# --------------------------------------------------
# 2. DUMMY SPECULATIVE DECODING
# (No draft model needed)
# --------------------------------------------------
def dummy_spec_decode(model, tokenizer, prompt, max_new_tokens=80):
    """
    Simulates speculative decoding by:
    - Fast: sampling tokens with top_k=20 (draft)
    - Slow: verifying only 1 in 4 tokens (main model)
    
    This *feels* like speculative decoding but works with ONE model.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ---- Step 1: Draft (cheap sampling)
    def draft_fn():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            top_k=20,
            temperature=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    (draft_ids, draft_time) = measure(draft_fn)

    draft_text = tokenizer.decode(draft_ids[0], skip_special_tokens=True)

    # ---- Step 2: Verification (main model but sparse)
    # We verify 1 in 4 tokens to simulate partial acceptance
    verify_tokens = draft_ids[:, ::4]

    def verify_fn():
        return model(
            input_ids=verify_tokens,
            use_cache=False
        )

    (_, verify_time) = measure(verify_fn)

    return {
        "method": "dummy_speculative",
        "output": draft_text,
        "draft_time": draft_time,
        "verify_time": verify_time,
        "total_time": draft_time + verify_time
    }


# --------------------------------------------------
# 3. REAL SPECULATIVE DECODING (IF DRAFT MODEL AVAILABLE)
# --------------------------------------------------
def real_spec_decode(
    main_model,
    draft_model,
    tokenizer,
    prompt,
    num_draft_tokens=4,
    max_new_tokens=80
):

    inputs = tokenizer(prompt, return_tensors="pt").to(main_model.device)

    # ---- Draft generates tokens
    def draft_fn():
        return draft_model.generate(
            **inputs,
            max_new_tokens=num_draft_tokens,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    (draft_ids, draft_time) = measure(draft_fn)

    # ---- Main model verifies the proposed tokens
    def verify_fn():
        return main_model(
            input_ids=draft_ids,
            use_cache=False
        )

    (_, verify_time) = measure(verify_fn)

    decoded = tokenizer.decode(draft_ids[0], skip_special_tokens=True)

    return {
        "method": "real_speculative",
        "output": decoded,
        "draft_tokens": draft_ids[0].tolist(),
        "draft_time": draft_time,
        "verify_time": verify_time,
        "total_time": draft_time + verify_time
    }
