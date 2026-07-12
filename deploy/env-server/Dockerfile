# packages the FastAPI RL Environment server (the "/reset", "/step",
# and "/grader" endpoints). Built and run alongside the Gradio demo
# Space container in docker-compose.

FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies for the FastAPI app only. Heavy ML deps
# (torch, transformers, trl, peft, bitsandbytes, accelerate) are
# intentionally NOT installed here — the env server doesn't load a
# model, it just hands out episodes and grades completions, so
# shipping a multi-GB ML image would be pure waste. The demo Space
# doesn't install them either (the Gradio client only HTTPs to us).
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Server code, environment module (Episode, SessionManager, grader,
# reward components), and the processed-episode dataset the /reset
# route reads on every call. Layered as separate COPYs so the
# dataset (the most frequently changing of the three) can be
# re-pushed without invalidating the pip-install layer.
COPY server/ server/
COPY src/ src/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
