# One image, several roles (api / grpc / worker / trainer), chosen by the command.
FROM python:3.13-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml ./
COPY driftline ./driftline
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.13-slim
# libgomp: LightGBM's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
WORKDIR /app
COPY scripts ./scripts
RUN mkdir -p /app/data && chown -R app:app /app
USER app
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 GIT_PYTHON_REFRESH=quiet DRIFTLINE_DATA_DIR=/app/data
EXPOSE 8000 50051
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs localhost:8000/health || exit 1
CMD ["driftline", "serve"]
