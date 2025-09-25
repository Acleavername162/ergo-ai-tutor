# Use GHCR mirror of the Python base image (avoids Docker Hub pulls)
FROM ghcr.io/library/python:3.11-slim

# Use bash with pipefail for safer installs
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

# -------- System deps --------
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# -------- Python deps --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- App code --------
COPY . .

# Make the system prompt path explicit (your code will auto-read this)
ENV ERGO_SYSTEM_PROMPT_PATH=/app/ergo_system_prompt.txt

# Ollama + tutor defaults (match your backend)
ENV OLLAMA_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M \
    OLLAMA_TIMEOUT=60 \
    OLLAMA_MAX_TOKENS=500 \
    OLLAMA_TEMP=0.7

# -------- Install Ollama in the image --------
RUN curl -fsSL https://ollama.com/install.sh | sh

# -------- Startup script --------
# Waits for Ollama, pulls model if needed, then launches your server
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
'' \
'# Start Ollama in background' \
'ollama serve &>/tmp/ollama.log &' \
'' \
'# Wait for Ollama HTTP to be available' \
'for i in {1..60}; do' \
'  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then' \
'    break' \
'  fi' \
'  sleep 1' \
'done' \
'' \
'# Pull the configured model (idempotent)' \
'MODEL="${OLLAMA_MODEL:-llama3.1:8b-instruct-q4_K_M}"' \
'ollama pull "$MODEL" || true' \
'' \
'# Launch FastAPI app (adjust if your entry is different)' \
'exec python3 server_platform.py' \
> /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

# Healthcheck against your /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
