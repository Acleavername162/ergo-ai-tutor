# Ergo AI Tutor API

FastAPI service for Project RISE that talks to **Ollama** (local LLM), with:
- Secure API key enforcement
- CORS controls
- SQLite persistence (simple bridge)
- Docker & Docker Compose support
- Per-user short conversation memory (configurable)

---

## Repository Layout
ergo-ai/
├─ server.py # FastAPI app (entrypoint)
├─ ergo_ai_tutor.py # Tutor + Ollama client (routing, retries, memory)
├─ platform_bridge.py # Synchronous SQLite helper
├─ ergo_system_prompt.txt # System prompt for Ergo
├─ requirements.txt # Full-stack Python deps
├─ Dockerfile # Full-stack build (cryptography/bcrypt ready)
├─ docker-compose.yml # Ollama + API stack
├─ .env.example # Copy to .env on server, set real secrets
├─ .gitignore
└─ .replit # Optional local run config for Replit
