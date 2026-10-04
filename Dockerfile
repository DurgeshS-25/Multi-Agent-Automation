# ---- Build a slim, production-ready image for the FastAPI app ----
FROM python:3.12-slim

# Don't buffer stdout/stderr (so logs appear immediately) and don't write
# .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (separate layer) so Docker caches them and
# doesn't reinstall on every code change — only when requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

# The API listens on 8000
EXPOSE 8000

# Run with uvicorn. No --reload in production (that's a dev-only convenience).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}