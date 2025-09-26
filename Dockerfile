version: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:11434/api/tags"]
      interval: 15s
      timeout: 5s
      retries: 20

  api:
    build: .
    container_name: ergo-api
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    # Load your local .env (not committed) to populate vars
    env_file:
      - .env
    environment:
      # Defaults if not present in .env
      HOST: ${HOST:-0.0.0.0}
      PORT: ${PORT:-8000}
      # Point API to the Ollama service by default
      OLLAMA_URL: ${OLLAMA_URL:-http://ollama:11434}
      OLLAMA_MODEL: ${OLLAMA_MODEL:-llama3.1:8b-instruct-q4_K_M}
      DEFAULT_MODEL: ${DEFAULT_MODEL:-llama3.1:8b-instruct-q4_K_M}
      OLLAMA_TIMEOUT: ${OLLAMA_TIMEOUT:-90}
      PLATFORM_DB_PATH: ${PLATFORM_DB_PATH:-/app/data/platform.db}
      DATABASE_URL: ${DATABASE_URL:-sqlite:////app/data/platform.db}
      ALLOWED_ORIGINS: ${ALLOWED_ORIGINS:-http://localhost:3000}
      API_KEY: ${API_KEY:-}
      JWT_SECRET_KEY: ${JWT_SECRET_KEY:-}
    volumes:
      - ./ergo_system_prompt.txt:/app/ergo_system_prompt.txt:ro
      - api_data:/app/data
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 20

volumes:
  ollama_models:
  api_data:
