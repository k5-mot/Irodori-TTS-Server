# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}
ARG IRODORI_TTS_BACKEND=cu128
ARG IRODORI_DOWNLOAD_SAMPLE_VOICES=true
ARG IRODORI_VOICE_SAMPLE_REPO=Aratako/Irodori-TTS-500M-v3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    IRODORI_BUNDLED_VOICES_DIR=/opt/irodori/voices

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-install-project --extra "${IRODORI_TTS_BACKEND}"

COPY README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-editable --extra "${IRODORI_TTS_BACKEND}"

RUN --mount=type=cache,target=/root/.cache/huggingface,sharing=locked \
    if [ "${IRODORI_DOWNLOAD_SAMPLE_VOICES}" = "true" ]; then \
        /app/.venv/bin/irodori-openai-tts-fetch-voices \
            --repo "${IRODORI_VOICE_SAMPLE_REPO}" \
            --voices-dir "${IRODORI_BUNDLED_VOICES_DIR}"; \
    fi

EXPOSE 8088

CMD ["/app/.venv/bin/python", "-m", "irodori_openai_tts"]
