FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create startup script
RUN echo '#!/bin/bash\nollama serve &\nsleep 10\nollama pull llama3.1\npython3 server_platform.py' > /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
