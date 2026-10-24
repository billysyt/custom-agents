FROM python:3.13.9-slim-bookworm@sha256:b685a4fa58bb19d1814d78a1ec0f0208f351452724f78b20212c984d6e124a34
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 LANGSMITH_TRACING=false DO_NOT_TRACK=1
WORKDIR /app
COPY requirements.lock ./
RUN pip install --disable-pip-version-check -r requirements.lock
COPY pyproject.toml README.md ./
COPY apps ./apps
COPY packages ./packages
COPY agents ./agents
RUN pip install --no-deps . && useradd --uid 10001 --create-home agent
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "custom_agents.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
