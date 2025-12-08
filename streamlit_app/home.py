import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.title("🚀 Inference Optimization Explorer")

st.markdown("""
Welcome to the **LLM Inference Optimization Dashboard**.

Choose an optimization method from the sidebar:

### Available Modules:
- **KV-Cache Optimization**
- **Speculative Decoding**
- **Batching / Throughput Scaling**
- **vLLM vs HF Transformers**
- (More coming…)

This structure allows the project to scale into a full research framework.
""")
