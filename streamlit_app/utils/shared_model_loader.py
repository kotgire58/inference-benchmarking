import yaml
from models.load_model import load_model
import streamlit as st

@st.cache_resource
def load_shared_model():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return load_model(cfg["model_name"], cfg["device"])
