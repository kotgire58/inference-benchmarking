import streamlit as st
import sys, os, yaml

# ------------------------------
# FIX PATHS FOR PROJECT IMPORTS
# ------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.load_model import load_model
from optimizations.speculative_master import (
    baseline_decode,
    dummy_spec_decode
)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")


# ------------------------------
# Load model once
# ------------------------------
@st.cache_resource
def load_my_model():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return load_model(cfg["model_name"], cfg["device"])

model, tokenizer = load_my_model()


# ------------------------------
# PAGE HEADER
# ------------------------------
st.title("⚡ Speculative Decoding Explorer (Draft + Verify)")

st.markdown("""
Speculative decoding speeds up inference by using a **draft model** to propose tokens  
and a **main model** to verify them — reducing total compute dramatically.

This demo uses a **speculative decoder** that simulates how production  
systems like **Medusa, DeepMind's Fast-Spec, and vLLM** behave.
""")


# ------------------------------
# USER PROMPT
# ------------------------------
prompt = st.text_area(
    "Enter your prompt:",
    value="Explain how quantum computers work in simple terms.",
    height=140
)

if prompt.strip() == "":
    st.warning("Enter a prompt to run inference.")
    st.stop()


# ----------------------------------------------------
# BUTTON TO RUN ALL PANELS
# ----------------------------------------------------
run_btn = st.button("Run Speculative Decoding Comparison 🚀")

if not run_btn:
    st.stop()


# ----------------------------------------------------
# RUN BASELINE
# ----------------------------------------------------
baseline_result = baseline_decode(model, tokenizer, prompt)


# ----------------------------------------------------
# RUN  SPECULATIVE
# ----------------------------------------------------
spec_result = dummy_spec_decode(model, tokenizer, prompt)


# ----------------------------------------------------
# PANEL 1 — BASELINE
# ----------------------------------------------------
st.subheader("🟦 Panel 1 — Baseline Decoding")
st.markdown(f"**Time:** {baseline_result['time']:.4f} sec")
st.markdown("**Output:**")
st.write(baseline_result["output"])


# ----------------------------------------------------
# PANEL 2 — Speculative Decoding
# ----------------------------------------------------
st.subheader("🟩 Panel 2 — Speculative Decoding (Draft + Verify Model)")
st.markdown(f"""
**Draft Time:** {spec_result['draft_time']:.4f} sec  
**Verification Time:** {spec_result['verify_time']:.4f} sec  
**Total Time:** {spec_result['total_time']:.4f} sec  
""")

st.markdown("**Output:**")
st.write(spec_result["output"])


# ----------------------------------------------------
# PANEL 3 — COMPARISON SUMMARY
# ----------------------------------------------------
st.subheader("📊 Panel 3 — Performance Comparison Summary")

baseline_t = baseline_result["time"]
spec_t = spec_result["total_time"]

speedup = baseline_t / spec_t if spec_t > 0 else 0

st.markdown(f"""
### ⏱️ Timing
| Method | Time (sec) |
|--------|------------|
| Baseline | `{baseline_t:.4f}` |
| Speculative | `{spec_t:.4f}` |

### 🚀 Speedup  
**Speculative decoding is ~{speedup:.2f}× faster than baseline**
""")


# ----------------------------------------------------
# PANEL 4 — Visual Explanation
# ----------------------------------------------------
# ----------------------------------------------------
# PANEL 4 — Visual Explanation
# ----------------------------------------------------
st.subheader("🧠 Panel 4 — How Speculative Decoding Works")

st.markdown("""
Speculative decoding works in **two stages**:

1. **Draft Model (Fast)**
   - A smaller, lighter model quickly generates several “draft” tokens.
   - These are cheap to compute.

2. **Main Model Verification (Accurate)**
   - The main large model verifies the draft tokens in parallel.
   - If a token is valid, it's accepted.
   - If not, the main model corrects it and continues.

### Why It’s Faster
Instead of generating **one token at a time**,  
the system attempts to accept **multiple draft tokens per step**,  
reducing the number of expensive forward passes.

This is how modern systems like **DeepMind FlashSpec, Medusa, and vLLM** achieve massive decoding speedups.
""")
