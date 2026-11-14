FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir "pydantic==2.14.3"

WORKDIR /app
COPY src/__init__.py            src/__init__.py
COPY src/models_agent/          src/models_agent/
COPY src/sandbox/               src/sandbox/

USER 65534:65534
ENTRYPOINT ["python", "-m", "src.sandbox.worker"]
