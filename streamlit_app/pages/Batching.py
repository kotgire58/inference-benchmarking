import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.title("📦 Batching & Throughput Explorer")
st.info("This will simulate large-batch inference scaling.")
