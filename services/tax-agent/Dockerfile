# ---------- Stage 1: builder ----------
FROM python:3.12-alpine AS builder

# Compilers for any package without a musllinux wheel. Costs build time
# only — none of this reaches the final image.
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
    && pip install --no-cache-dir --no-compile -r requirements.txt

# Shrink the venv before it is copied into the runtime stage.
# `;` not `&&` so a no-op does not abort the build.
RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true; \
    find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    find /opt/venv -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true


# ---------- Stage 2: runtime ----------
FROM python:3.12-alpine AS runtime

RUN apk add --no-cache libgcc libstdc++

RUN adduser -D -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=appuser:appuser app/ ./app/

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O- http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--no-server-header", "--timeout-keep-alive", "65"]