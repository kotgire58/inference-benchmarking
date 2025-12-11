---
title: "Inference Benchmarking Suite"
emoji: "⚡"
colorFrom: "blue"
colorTo: "purple"
sdk: "streamlit"
sdk_version: "1.35.0"
app_file: "streamlit_app/Intro.py"
pinned: false
---


🚀 Inference Benchmarking Suite (Streamlit Edition)

A modular, educational benchmarking toolkit designed to teach and demonstrate modern LLM inference optimizations — including batching, KV-cache reuse, speculative decoding, vLLM acceleration, and real-time chatbot comparisons — with a clean Streamlit UI.

This project is structured for hands-on exploration, making it ideal for learning how different inference strategies impact:

⚡ Latency

🔥 Throughput (tokens/sec)

💾 Memory usage

💵 Cost efficiency

🤖 Real-time user experience

📁 Project Structure
INFERENCE-BENCHMARKING/
│
├── benchmarks/                 # Core benchmarking utilities
│── models/                     # Model loading & backend wrappers
│── optimizations/              # Optional custom optimization modules
│
├── streamlit_app/              # Main Streamlit application
│   ├── Intro.py                # Home page (entry screen)
│   ├── config.toml             # Streamlit multipage configuration
│   └── pages/
│       ├── Batching.py         # Batching vs non-batching demo
│       ├── ChatbotDemo.py      # Chat inference comparison
│       ├── FinalBenchmark.py   # Unified benchmark runner
│       ├── kv_cache.py         # KV-cache speedup visualization
│       ├── SpeculativeDecoding.py # Draft model vs target model
│       └── VLLM.py             # vLLM-specific benchmark page
│
├── utils/                      # Shared helpers for timing, logging, etc.
│
├── run_benchmark.py            # CLI runner for benchmarking
├── requirements.txt
├── Dockerfile
└── README.md                   # (this file)

🎯 What This App Teaches

Each page inside the Streamlit UI focuses on one inference concept, showing both code and performance impact:

🧩 1. Batching

Demonstrates how batching multiple prompts drastically increases throughput.

Shows tokens/sec vs batch size.

💡 2. KV-Cache

Visualizes how reusing cached key/value tensors reduces decoding cost.

Demonstrates "streaming-like" speedup.

⚡ 3. Speculative Decoding

Draft model generates N tokens → target model verifies.

Shows latency reduction %.

🔥 4. vLLM Engine

Compares vanilla inference vs paged attention.

Great for understanding GPU memory efficiency.

🤖 5. Chatbot Demo

Side-by-side inference comparison.

Helps visualize real-time responsiveness differences.

📊 6. Final Benchmark

A clean, unified benchmark runner measuring:

Latency

Throughput

Cost estimates

Stability across multiple runs

▶️ Running Locally (Recommended for Testing)
1. Install dependencies
pip install -r requirements.txt

2. Launch the Streamlit app
streamlit run streamlit_app/Intro.py


You should now see the multipage UI load with all benchmark pages.