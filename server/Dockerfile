FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies (no dev extras)
RUN uv pip install --system --no-cache -e ".[core]" 2>/dev/null || \
    uv pip install --system --no-cache \
        openenv-core \
        fastapi \
        "uvicorn[standard]" \
        websockets \
        pydantic \
        torch \
        transformers \
        huggingface-hub \
        numpy \
        scikit-learn \
        python-dotenv \
        pyyaml

# Copy the full project
COPY . .

# Environment variables (overridden at runtime)
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HF_HOME=/app/.cache/huggingface

# Pre-download ESM2 weights at build time so container starts fast
RUN python3 -c "\
from transformers import EsmTokenizer, EsmModel; \
EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
print('ESM2 weights cached.')"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
