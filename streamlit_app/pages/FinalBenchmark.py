import streamlit as st
import sys, os, yaml
import pandas as pd

# ------------------------------
# FIX PATHS FOR PROJECT IMPORTS
# ------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.load_model import load_model
from benchmarks.baseline import run_baseline
from optimizations.kv_cache_master import generate_with_kv_cache
from optimizations.batching_master import (
    sequential_inference,
    static_batch_inference,
    dynamic_batch_inference,
    prefill_merge_inference
)
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
st.title("🏁 Final LLM Benchmark Comparison Suite")

st.markdown("""
This page compares **all optimization techniques** side-by-side:

- **Baseline**
- **KV-Cache Variants:** Small, Optimal, Large, Custom
- **Batching:** Static, Dynamic, Prefill-Merge
- **Speculative Decoding (Draft + Verify)**

Each method runs on the *same prompt* and results appear in a clean grid.
""")


# ------------------------------
# PROMPT INPUT
# ------------------------------
prompt = st.text_area(
    "Enter a prompt for benchmarking:",
    value="Explain quantum computers in simple words.",
    height=140
)

if prompt.strip() == "":
    st.warning("Please enter a prompt.")
    st.stop()


# ------------------------------
# METHOD SELECTION
# ------------------------------
st.sidebar.title("Select Methods to Benchmark")

baseline_on = st.sidebar.checkbox("Baseline", True)

kv_small_on   = st.sidebar.checkbox("KV Small (16 tokens)")
kv_opt_on     = st.sidebar.checkbox("KV Optimal (full)")
kv_large_on   = st.sidebar.checkbox("KV Large (4× inflated)")
kv_custom_on  = st.sidebar.checkbox("KV Custom")

custom_size = None
if kv_custom_on:
    custom_size = st.sidebar.slider("Custom KV-Cache Size", 1, 2048, 256)


batch_static_on   = st.sidebar.checkbox("Static Batch")
batch_dynamic_on  = st.sidebar.checkbox("Dynamic Batch")
batch_prefill_on  = st.sidebar.checkbox("Prefill-Merge Batch")

speculative_on    = st.sidebar.checkbox("Speculative Decoding")


enabled_methods = [
    m for m, flag in {
        "Baseline": baseline_on,
        "KV Small": kv_small_on,
        "KV Optimal": kv_opt_on,
        "KV Large": kv_large_on,
        "KV Custom": kv_custom_on,
        "Static Batch": batch_static_on,
        "Dynamic Batch": batch_dynamic_on,
        "Prefill-Merge": batch_prefill_on,
        "Speculative": speculative_on,
    }.items() if flag
]


# ------------------------------
# RUN BUTTON
# ------------------------------
run_btn = st.button("Run All Selected Benchmarks 🚀")

if not run_btn:
    st.stop()


# ------------------------------
# RUN BENCHMARKS
# ------------------------------
results = {}
baseline_time = None  # used for speedup calculations

for method in enabled_methods:

    # ---------------- Baseline ----------------
    if method == "Baseline":
        output, t = run_baseline(model, tokenizer, prompt, max_new_tokens=80)
        baseline_time = t
        results[method] = {
            "output": output,
            "total_time": t,
        }

    # ---------------- KV Small ----------------
    elif method == "KV Small":
        res = generate_with_kv_cache(model, tokenizer, prompt, 80, cache_mode="small")
        results[method] = res

    # ---------------- KV Optimal ----------------
    elif method == "KV Optimal":
        res = generate_with_kv_cache(model, tokenizer, prompt, 80, cache_mode="optimal")
        results[method] = res

    # ---------------- KV Large ----------------
    elif method == "KV Large":
        res = generate_with_kv_cache(model, tokenizer, prompt, 80, cache_mode="large")
        results[method] = res

    # ---------------- KV Custom ----------------
    elif method == "KV Custom":
        res = generate_with_kv_cache(
            model, tokenizer, prompt, 80,
            cache_mode="custom",
            custom_cache_size=custom_size
        )
        results[method] = res

    # ---------------- Static Batch ----------------
    elif method == "Static Batch":
        res = static_batch_inference(model, tokenizer, [prompt])
        results[method] = {
            "output": res["outputs"][0],
            "total_time": res["total_time"],
        }

    # ---------------- Dynamic Batch ----------------
    elif method == "Dynamic Batch":
        res = dynamic_batch_inference(model, tokenizer, [prompt])
        results[method] = {
            "output": res["outputs"][0],
            "total_time": res["total_time"],
        }

    # ---------------- Prefill-Merge ----------------
    elif method == "Prefill-Merge":
        res = prefill_merge_inference(model, tokenizer, [prompt])
        results[method] = {
            "output": res["outputs"][0],
            "total_time": res["total_time"],
        }

    # ---------------- Speculative ----------------
    elif method == "Speculative":
        res = dummy_spec_decode(model, tokenizer, prompt)
        results[method] = {
            "output": res["output"],
            "total_time": res["total_time"],
        }


# ------------------------------
# DISPLAY GRID OF RESULTS
# ------------------------------
st.header("📊 Side-by-Side Benchmark Results")

cols = st.columns(len(enabled_methods))

for col, method in zip(cols, enabled_methods):
    with col:
        st.subheader(method)

        # Output (truncated)
        out = results[method]["output"]
        if len(out) > 200:
            st.write(out[:200] + "...")
            with st.expander("Show full output"):
                st.write(out)
        else:
            st.write(out)

        # Time
        t = results[method]["total_time"]
        st.write(f"**Total Time:** `{t:.4f}` sec")

        # Speedup
        if baseline_time:
            speed = baseline_time / t if t > 0 else 0
            st.write(f"**Speedup:** `{speed:.2f}×`")


# ------------------------------
# SUMMARY TABLE
# ------------------------------
st.header("📌 Summary Table")

df = pd.DataFrame([
    {
        "Method": method,
        "Total Time (s)": results[method]["total_time"],
        "Speedup vs Baseline": (baseline_time / results[method]["total_time"])
        if baseline_time else None
    }
    for method in enabled_methods
])

st.dataframe(df, use_container_width=True)


# ------------------------------
# BAR CHART
# ------------------------------
st.header("📈 Time Comparison Chart")
chart_df = df.set_index("Method")["Total Time (s)"]
st.bar_chart(chart_df)
