FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . /app/

# Expose port used by Streamlit
EXPOSE 7860

# Streamlit needs these or it crashes on HF Spaces
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# ENTRY POINT — THIS RUNS YOUR Intro.py
CMD ["streamlit", "run", "streamlit_app/Intro.py", "--server.port=7860", "--server.address=0.0.0.0"]
