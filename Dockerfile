# syntax=docker/dockerfile:1.7
# ---- build: wheel ------------------------------------------------------------------------------
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md LICENSE CHANGELOG.md Makefile ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ---- runtime -----------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SQLPILOT_MEMORY_URL=sqlite:////home/sqlpilot/.sqlpilot/memory.db
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin sqlpilot
COPY --from=build /dist/*.whl /tmp/
RUN pip install "$(ls /tmp/sqlpilot-*.whl)[server,postgres,mcp,otel]" && rm -f /tmp/*.whl
USER sqlpilot
WORKDIR /home/sqlpilot
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["sqlpilot"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
