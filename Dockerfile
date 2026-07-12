FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn pydantic gradio requests
# Hugging Face exposes port 7860 by default
EXPOSE 7860
# Start FastAPI in the background, wait 3 seconds, then start Gradio
CMD uvicorn server.main:app --host 0.0.0.0 --port 8000 & \
    sleep 3 && \
    python deploy/demo-space/app.py
