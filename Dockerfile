FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Only what the FastAPI app reads at runtime: the backend code (API + static
# dashboard) and the precomputed results it serves. Raw C-MAPSS data and
# model checkpoints are only used by the offline ml/ training scripts, never
# imported by the running app, so they're left out of the image.
COPY backend/ ./backend/
COPY models/results/ ./models/results/

ENV PYTHONUNBUFFERED=1

# Cloud Run sets $PORT; uvicorn must bind to it, not a hardcoded port.
# Shell form (with `exec`) is required so ${PORT} expands and uvicorn still
# becomes PID 1 (proper SIGTERM handling on Cloud Run scale-down).
CMD exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8080}
