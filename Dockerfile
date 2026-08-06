FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system syco \
    && useradd --system --gid syco --create-home syco

WORKDIR /app
COPY deploy/demo-space/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --requirement requirements.txt
COPY --chown=syco:syco deploy/demo-space/app.py ./app.py

USER syco
EXPOSE 7860
CMD ["python", "app.py"]
