import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def count_tokens(tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return len(tokens)


def get_attention_heatmap(model, tokenizer, prompt, past_key_values=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            past_key_values=past_key_values
        )

    # attentions: list of layers → heads → seq → seq
    attentions = outputs.attentions

    heatmaps = []
    for att in attentions:
        # att is shape (1, heads, seq_q, seq_k)

        # average across heads
        layer_att = att[0].mean(dim=0)

        # 🔥 FIX: Convert BF16 → FP32 BEFORE converting to NumPy
        layer_att = layer_att.to(torch.float32)

        heatmaps.append(layer_att.cpu().numpy())

    return heatmaps


def plot_attention_heatmap(matrix, title="Attention"):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    return buf.getvalue()
