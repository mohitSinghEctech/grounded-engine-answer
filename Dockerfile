# ---------- Stage 1: builder ----------
FROM python:3.12-alpine AS builder

# Compilers for any package lacking a musllinux wheel (likely uvloop/httptools).
# Costs build TIME only — none of this reaches the final image.
RUN apk add --no-cache \
        build-base \
        python3-dev \
        libffi-dev \
        musl-dev

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Strip pip/setuptools from the runtime venv — not needed once installed.
RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true


# ---------- Stage 2: runtime ----------
FROM python:3.12-alpine AS runtime

# Runtime shared libs for the compiled extensions built above.
RUN apk add --no-cache libgcc libstdc++

RUN adduser -D -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=appuser:appuser app/ ./app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--no-server-header", "--timeout-keep-alive", "65"]