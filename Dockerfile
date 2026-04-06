FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install all dependencies
RUN uv pip install --system --no-cache \
    "openenv-core>=0.2.1" \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    "pydantic>=2.7.0" \
    torch \
    transformers \
    huggingface-hub \
    numpy \
    scikit-learn \
    python-dotenv \
    pyyaml

# Copy entire project
COPY . .

ENV PYTHONPATH=/app
ENV PORT=7860
ENV HF_HOME=/app/.cache/huggingface

# Pre-download ESM2 weights at build time
RUN python3 -c "\
from transformers import EsmTokenizer, EsmModel; \
EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
print('ESM2 weights cached.')"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
