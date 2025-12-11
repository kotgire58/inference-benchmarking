import streamlit as st

st.set_page_config(page_title="HF Test", page_icon="🧪")

st.title("🧪 Hugging Face Space Test")
st.write("If you can see this, your Dockerfile + CMD + PORT setup is working.")

# Show environment variable PORT (for debugging)
import os
st.write("PORT:", os.getenv("PORT"))

st.success("Streamlit is running successfully!")
