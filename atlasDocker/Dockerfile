FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

RUN useradd -m -u 1000 user \
 && mkdir -p /app/.cache/huggingface \
 && chown -R 1000:1000 /app /app/.cache
USER user
WORKDIR /app

COPY --chown=user app.py app.py
CMD ["/bin/sh", "-c", "python app.py && exec embedding-atlas embeddings.parquet --vector embedding --text text --host 0.0.0.0 --port 7860"]
# port 7860 for embedding-atlas web server
# HF_TOKEN would need to be set as an environment variable