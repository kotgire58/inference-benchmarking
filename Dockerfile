FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Streamlit config for Hugging Face Spaces
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create config file
RUN mkdir -p ~/.streamlit
RUN echo '[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 7860\n\
address = "0.0.0.0"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "light"' > ~/.streamlit/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Use the PORT environment variable with a fallback
CMD ["sh", "-c", "streamlit run streamlit_app/Intro.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]