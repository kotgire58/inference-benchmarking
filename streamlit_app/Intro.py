import streamlit as st
import sys, os
import psutil
import GPUtil
from streamlit_autorefresh import st_autorefresh

# Allow imports from root of repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

st.set_page_config(page_title="Inference Optimization Explorer", layout="wide")


# --------------------------------------------------------
# PAGE HEADER
# --------------------------------------------------------
st.title("🚀 Inference Optimization Explorer")

st.markdown("""
Welcome to your **LLM Inference Optimization Dashboard**!  
This interactive toolkit demonstrates how real production inference engines like **vLLM**,  
**HuggingFace Transformers**, and custom servers optimize **latency, throughput, and memory**.

Use the sidebar to explore:

### 🔧 Available Modules
- **KV-Cache Optimization**
- **Speculative Decoding**
- **Batching & Throughput Scaling**
- **Final Chatbot Demo**
- *(More expansions ready — FlashAttention, Quantization, MoE, vLLM, etc.)*

---
""")


# --------------------------------------------------------
# 🔥 LIVE SYSTEM RESOURCE MONITOR (Refreshes Every 3s)
# --------------------------------------------------------
@st.fragment
def live_system_stats():
    st_autorefresh(interval=3000, key="live_stats_refresh")

    st.subheader("🖥️ Live System Resource Usage")

    col1, col2, col3 = st.columns(3)

    # CPU Usage
    cpu = psutil.cpu_percent()
    col1.metric("CPU Usage", f"{cpu:.1f}%")

    # RAM Usage
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024**3)
    ram_total = ram.total / (1024**3)
    col2.metric("RAM Usage", f"{ram.percent:.1f}%", f"{ram_used:.2f} GB / {ram_total:.2f} GB")

    # GPU Usage
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_used = gpu.memoryUsed / 1024
        gpu_total = gpu.memoryTotal / 1024

        col3.metric(
            "GPU Memory",
            f"{gpu.memoryUtil * 100:.1f}%",
            f"{gpu_used:.2f} GB / {gpu_total:.2f} GB",
        )
    else:
        col3.metric("GPU Memory", "N/A", "No GPU Detected")


live_system_stats()

st.markdown("---")


# --------------------------------------------------------
# FEATURE DESCRIPTIONS
# --------------------------------------------------------
st.subheader("📦 What You Can Explore Here")

colA, colB = st.columns(2)

with colA:
    st.markdown("""
    ### ⚡ KV-Cache Optimization  
    Learn how trimming, inflating, or customizing the KV-cache  
    affects **prefill**, **decode latency**, and **hallucination risk**.

    ### 🔀 Speculative Decoding  
    See how draft-model token proposals get verified by the main model  
    for **2–4× faster decoding**.

    ### 👥 Batching Strategies  
    - Sequential vs Static Batch  
    - Dynamic Batching (production-style queueing)  
    - Prefill-Merge (vLLM technique)
    """)

with colB:
    st.markdown("""
    ### 🤖 Final Chatbot Demo  
    A ChatGPT-style assistant where you can switch inference modes in real time.

    ### 🔍 Academic + Industry Learning  
    This dashboard is structured exactly like real inference benchmarks used at:
    - Anthropic  
    - OpenAI  
    - Google DeepMind  
    - NVIDIA  
    - vLLM Research Group  
    """)

st.markdown("---")

st.info("""
You're now ready to explore each module using the **sidebar**!
""")
