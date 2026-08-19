# Disposable evaluator image.  It contains no model, credentials, repository,
# or generated programs.  Candidate code enters only through a purpose-built
# temporary workspace at invocation time.
FROM python:3.12-slim@sha256:REPLACE_WITH_REVIEWED_DIGEST

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/evalplus-home \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 65532 --shell /usr/sbin/nologin evaluator \
    && python -m pip install --no-cache-dir --no-compile 'evalplus==0.3.1' \
    && rm -rf /root/.cache /tmp/*

USER evaluator
WORKDIR /work

ENTRYPOINT ["python", "-m", "evalplus.evaluate"]
