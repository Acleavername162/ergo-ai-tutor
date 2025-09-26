# Full-stack Dockerfile for Ergo API (with deps for cryptography/bcrypt)
FROM python:3.11-slim

WORKDIR /app

# --- System dependencies for wheels/builds ---
# - build-essential: gcc/make for building native extensions
# - libssl-dev, libffi-dev: required by cryptography / bcrypt
# - curl: used by healthchecks or debugging
# - pkg-config: safer builds for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev pkg-config curl \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App files ---
COPY server.py ergo_ai_tutor.py platform_bridge.py ergo_system_prompt.txt ./

# --- Environment defaults (override with .env or compose) ---
ENV HOST=0.0.0.0
ENV PORT=8000
# If running with docker-compose + Ollama service, override to http://ollama:11434
ENV OLLAMA_URL=http://127.0.0.1:11434
ENV OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
ENV DEFAULT_MODEL=llama3.1:8b-instruct-q4_K_M
ENV OLLAMA_TIMEOUT=90

# --- Expose API port ---
EXPOSE 8000

# --- Start the API ---
CMD ["python", "server.py"]
