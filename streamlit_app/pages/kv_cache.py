import streamlit as st
import sys, os
import yaml

# --------------------------------------------------------
# FIX PATHS — MAKE PROJECT ROOT ACCESSIBLE
# --------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

# --------------------------------------------------------
# FIX UTILS PATH — MUST COME BEFORE IMPORTS
# --------------------------------------------------------
UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/"))
sys.path.insert(0, UTILS_PATH)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")

# --------------------------------------------------------
# IMPORT AFTER PATHS ARE FIXED
# --------------------------------------------------------
from models.load_model import load_model
from benchmarks.baseline import run_baseline
from optimizations.kv_cache_master import generate_with_kv_cache
from utils.shared_models_utils import get_attention_heatmap, plot_attention_heatmap



# ============================================================
# LOAD MODEL ONCE (CACHED)
# ============================================================
@st.cache_resource
def load_my_model():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model(cfg["model_name"], cfg["device"])
    return model, tokenizer

model, tokenizer = load_my_model()


# ============================================================
# PAGE UI
# ============================================================
st.title("🔍 KV-Cache Optimization Explorer")

st.markdown("""
Explore how KV-cache manipulation affects:
- 🚀 Inference latency  
- 🧠 Attention patterns  
- 📉 Output quality  
- 🪣 Context retention  
""")


# ============================================================
# SIDEBAR PANELS
# ============================================================
comparison_panels = st.sidebar.multiselect(
    "Enable Comparison Panels:",
    ["Panel 1", "Panel 2", "Panel 3", "Panel 4"],
    default=["Panel 1", "Panel 2", "Panel 3"]
)


# ============================================================
# USER PROMPT
# ============================================================
prompt = st.text_area(
    "Enter your prompt:",
    value="Explain quantum computing in simple words.",
    height=140
)


# ============================================================
# PANEL SETTINGS FUNCTION
# ============================================================
def render_panel(panel_name):
    st.subheader(f"📌 {panel_name}")

    mode = st.selectbox(
        f"KV-Cache Mode for {panel_name}:",
        ["Baseline", "Small Cache", "Optimal Cache", "Large Cache", "Custom Cache"],
        key=f"mode_{panel_name}"
    )

    explanation = {
        "Baseline": "Normal model behavior — no KV manipulation.",
        "Small Cache": "Only last **16 tokens** kept. Very fast, but loses context.",
        "Optimal Cache": "KV cache = full prompt tokens. Fast & accurate.",
        "Large Cache": "Artificially enlarged KV cache — shows slowdown.",
        "Custom Cache": "Specify an exact KV window size to mimic partial truncation."
    }

    st.info(explanation[mode])

    custom_size = None
    if mode == "Custom Cache":
        custom_size = st.slider(
            f"Custom KV cache size ({panel_name})",
            1, 2048, 256,
            key=f"custom_{panel_name}"
        )

    return {"mode": mode, "custom": custom_size}


# ============================================================
# RENDER SETTINGS FOR ALL PANELS
# ============================================================
panel_settings = {}
cols = st.columns(len(comparison_panels))

for col, panel_name in zip(cols, comparison_panels):
    with col:
        panel_settings[panel_name] = render_panel(panel_name)


# ============================================================
# TOKEN VISUALIZATION
# ============================================================
prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

def compute_effective_cache_size(mode, custom_size=None):
    if mode == "Small Cache":
        return 16
    if mode == "Optimal Cache":
        return prompt_tokens
    if mode == "Large Cache":
        return min(prompt_tokens * 4, 4096)
    if mode == "Custom Cache":
        return min(custom_size, prompt_tokens)
    return prompt_tokens  # baseline


st.subheader("📊 Token Visualization (KV Retention)")

cols = st.columns(len(comparison_panels))

for col, panel_name in zip(cols, comparison_panels):
    settings = panel_settings[panel_name]
    effective_cache = compute_effective_cache_size(settings["mode"], settings["custom"])

    with col:
        st.markdown(f"### {panel_name}")
        st.write(f"Prompt Tokens: **{prompt_tokens}**")
        st.write(f"Tokens Kept in KV Cache: **{effective_cache}**")
        st.write(f"Tokens Discarded: **{max(0, prompt_tokens - effective_cache)}**")
        st.progress(min(effective_cache / prompt_tokens, 1.0))


# ============================================================
# RUN BUTTON
# ============================================================
run_button = st.button("Run Comparison 🚀")


# ============================================================
# INFERENCE RESULTS
# ============================================================
if run_button:
    result_cols = st.columns(len(comparison_panels))

    for col, panel_name in zip(result_cols, comparison_panels):
        with col:
            st.markdown(f"## {panel_name} Results")

            mode = panel_settings[panel_name]["mode"]
            custom = panel_settings[panel_name]["custom"]

            if mode == "Baseline":
                output, total = run_baseline(model, tokenizer, prompt, max_new_tokens=80)
                st.write(output)
                st.write(f"**Total Time:** {total:.4f}s")

            else:
                cache_mode = (
                    "small" if mode == "Small Cache"
                    else "optimal" if mode == "Optimal Cache"
                    else "large" if mode == "Large Cache"
                    else "custom"
                )

                result = generate_with_kv_cache(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=80,
                    cache_mode=cache_mode,
                    custom_cache_size=custom
                )

                st.write(result["output"])
                st.write(
                    f"""
                    **Prefill:** {result['prefill_time']:.4f}s  
                    **Decode:** {result['decode_time']:.4f}s  
                    **Total:** {result['total_time']:.4f}s  
                    """
                )


# ============================================================
# ATTENTION HEATMAP
# ============================================================
st.subheader("🧠 Attention Heatmap Visualization")

heatmap_panel = st.selectbox(
    "Select which panel's attention to visualize:",
    comparison_panels
)

if st.button("Generate Attention Heatmaps 🧠"):

    settings = panel_settings[heatmap_panel]
    mode = settings["mode"]
    custom = settings["custom"]

    # Run model to get baseline attention + past cache
    outputs = model(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        use_cache=True,
        output_attentions=True
    )

    pkv = outputs.past_key_values

    # Apply truncation
    if mode == "Small Cache":
        new_pkv = []
        for k, v in pkv:
            new_pkv.append((k[:, :, -16:, :], v[:, :, -16:, :]))
        pkv = tuple(new_pkv)

    elif mode == "Custom Cache" and custom is not None:
        new_pkv = []
        for k, v in pkv:
            new_pkv.append((k[:, :, -custom:, :], v[:, :, -custom:, :]))
        pkv = tuple(new_pkv)

    # Get attention maps
    heatmaps = get_attention_heatmap(model, tokenizer, prompt, pkv)

    # Display first 4 layers
    cols = st.columns(4)
    for i, hm in enumerate(heatmaps[:4]):
        with cols[i]:
            png = plot_attention_heatmap(hm, f"Layer {i}")
            st.image(png, caption=f"Layer {i}", use_column_width=True)
