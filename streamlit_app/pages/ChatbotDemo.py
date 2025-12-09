import streamlit as st
import sys, os, yaml
import time

# ----------------------------------
# FIX PATHS FOR PROJECT IMPORTS
# ----------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.load_model import load_model
from benchmarks.baseline import run_baseline
from optimizations.kv_cache_master import generate_with_kv_cache
from optimizations.batching_master import static_batch_inference
from optimizations.speculative_master import (
    baseline_decode,
    dummy_spec_decode
)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")


# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_my_model():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return load_model(cfg["model_name"], cfg["device"])

model, tokenizer = load_my_model()


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(page_title="Chatbot Demo", layout="wide")

st.title("🤖 Final Chatbot Demo — With Optimization Modes")
st.markdown("""
This is the **final showcase chatbot** for your entire benchmarking project.  
It behaves like a modern conversational assistant while using different optimization strategies.

""")

# ----------------------------------
# INIT CHAT HISTORY
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# SIDEBAR — Optimization Selector
# ----------------------------------
st.sidebar.title("⚙️ Optimization Mode")

mode = st.sidebar.selectbox(
    "Choose inference mode:",
    [
        "Baseline",
        "KV Small (16 tokens)",
        "KV Optimal",
        "KV Large (inflated)",
        "KV Custom",
        "Speculative Decoding (dummy)"
    ]
)

custom_kv = None
if mode == "KV Custom":
    custom_kv = st.sidebar.slider(
        "Custom KV Cache Size",
        1, 2048, 256
    )


# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []


# ----------------------------------
# CHAT DISPLAY (ChatGPT-like bubbles)
# ----------------------------------
def render_chat():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align: right; margin: 8px 0;'>
                    <div style='display: inline-block; background:#2b7cff; padding:10px 14px; border-radius:12px; color:white; max-width:70%;'>
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='text-align: left; margin: 8px 0;'>
                    <div style='display: inline-block; background:#444; padding:10px 14px; border-radius:12px; color:white; max-width:70%;'>
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Timing info
            if "timing" in msg:
                st.markdown(
                    f"""
                    <div style='text-align:left; color:#aaa; font-size:13px; margin-top:-6px;'>
                        ⏱ {msg["timing"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


render_chat()


# ----------------------------------
# INPUT BOX
# ----------------------------------
# ----------------------------------
# CHAT INPUT BOX (with form)
# ----------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your message:",
        key="chat_input"
    )
    send = st.form_submit_button("Send")

if send and user_input.strip() != "":
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ------------------------------
    # RUN MODEL BASED ON MODE
    # ------------------------------
    if mode == "Baseline":
        out, t = run_baseline(model, tokenizer, user_input, max_new_tokens=150)
        response = out
        timing_str = f"Baseline | Total: {t:.4f}s"

    elif mode == "KV Small (16 tokens)":
        res = generate_with_kv_cache(model, tokenizer, user_input, 150, cache_mode="small")
        response = res["output"]
        timing_str = (
            f"KV Small | Prefill: {res['prefill_time']:.4f}s | "
            f"Decode: {res['decode_time']:.4f}s | Total: {res['total_time']:.4f}s"
        )

    elif mode == "KV Optimal":
        res = generate_with_kv_cache(model, tokenizer, user_input, 150, cache_mode="optimal")
        response = res["output"]
        timing_str = (
            f"KV Optimal | Prefill: {res['prefill_time']:.4f}s | "
            f"Decode: {res['decode_time']:.4f}s | Total: {res['total_time']:.4f}s"
        )

    elif mode == "KV Large (inflated)":
        res = generate_with_kv_cache(model, tokenizer, user_input, 150, cache_mode="large")
        response = res["output"]
        timing_str = (
            f"KV Large | Prefill: {res['prefill_time']:.4f}s | "
            f"Decode: {res['decode_time']:.4f}s | Total: {res['total_time']:.4f}s"
        )

    elif mode == "KV Custom":
        res = generate_with_kv_cache(
            model, tokenizer, user_input, 150,
            cache_mode="custom", custom_cache_size=custom_kv
        )
        response = res["output"]
        timing_str = (
            f"KV Custom ({custom_kv}) | Prefill: {res['prefill_time']:.4f}s | "
            f"Decode: {res['decode_time']:.4f}s | Total: {res['total_time']:.4f}s"
        )

    elif mode == "Speculative Decoding (dummy)":
        res = dummy_spec_decode(model, tokenizer, user_input)
        response = res["output"]
        timing_str = (
            f"Speculative (Dummy) | Draft: {res['draft_time']:.4f}s | "
            f"Verify: {res['verify_time']:.4f}s | Total: {res['total_time']:.4f}s"
        )

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timing": timing_str
    })

    st.rerun()
