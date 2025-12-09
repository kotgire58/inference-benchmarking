import streamlit as st
import sys, os, yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.load_model import load_model
from optimizations.batching_master import (
    sequential_inference,
    static_batch_inference,
    dynamic_batch_inference,
    prefill_merge_inference
)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_my_model():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return load_model(cfg["model_name"], cfg["device"])

model, tokenizer = load_my_model()


# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("🧩 Batching Optimization Explorer")

st.markdown("""
Explore how batching improves LLM throughput:

- **Sequential inference** → slowest, one-by-one  
- **Static batching** → pad + batch all prompts  
- **Dynamic batching** → group requests in real-time  
- **Prefill merge (vLLM style)** → prefill separately, decode together  

Use this page to understand *how real LLM servers achieve massive speedups*.
""")


# --------------------------------------------------
# PROMPT INPUT SECTION
# --------------------------------------------------
if "prompts" not in st.session_state:
    st.session_state.prompts = [""]


def add_prompt():
    st.session_state.prompts.append("")


def remove_prompt():
    if len(st.session_state.prompts) > 1:
        st.session_state.prompts.pop()


st.subheader("📝 Enter Prompts for Comparison")

for i, text in enumerate(st.session_state.prompts):
    st.session_state.prompts[i] = st.text_area(
        f"Prompt {i+1}",
        value=text,
        key=f"prompt_{i}"
    )

colA, colB = st.columns(2)
with colA:
    st.button("➕ Add Prompt", on_click=add_prompt)
with colB:
    st.button("➖ Remove Prompt", on_click=remove_prompt)

prompts = [p for p in st.session_state.prompts if p.strip()]

if not prompts:
    st.warning("Enter at least one prompt to run batching.")
    st.stop()


# --------------------------------------------------
# RENDER RESULT BEAUTIFULLY (NOT JSON)
# --------------------------------------------------
def render_result(title, result):
    st.markdown(f"## {title}")

    # Handle errors
    if "error" in result:
        st.error(result["error"])
        return

    # ---- METRICS ----
    if "total_time" in result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time", f"{result['total_time']:.4f}s")
        if "prefill_time" in result:
            col2.metric("Prefill Time", f"{result['prefill_time']:.4f}s")
        if "decode_time" in result:
            col3.metric("Decode Time", f"{result['decode_time']:.4f}s")

    # ---- RESPONSES ----
    st.markdown("### 📤 Outputs")
    outputs = result.get("outputs") or \
              [item["output"] for item in result.get("per_prompt", [])]

    for i, text in enumerate(outputs):
        with st.expander(f"Response {i+1}"):
            st.write(text)

    # ---- TECHNICAL DETAILS ----
    with st.expander("🔍 Technical Details"):
        st.json(result)


# --------------------------------------------------
# RUN BUTTONS
# --------------------------------------------------
st.subheader("⚙️ Run Batching Methods")

run_seq = st.button("Run Sequential")
run_static = st.button("Run Static Batch")
run_dynamic = st.button("Run Dynamic Batch")
run_prefill = st.button("Run Prefill Merge")
run_all = st.button("Run All Methods 🚀")


# --------------------------------------------------
# EXECUTION + UI RENDER
# --------------------------------------------------
if run_seq or run_all:
    result = sequential_inference(model, tokenizer, prompts)
    render_result("🐢 Sequential Inference", result)

if run_static or run_all:
    result = static_batch_inference(model, tokenizer, prompts)
    render_result("📦 Static Batch Inference", result)

if run_dynamic or run_all:
    result = dynamic_batch_inference(model, tokenizer, prompts)
    render_result("⚡ Dynamic Batch Inference", result)

if run_prefill or run_all:
    result = prefill_merge_inference(model, tokenizer, prompts)
    render_result("🚀 Prefill Merge (vLLM-Style)", result)
