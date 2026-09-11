# GPU image for the VLM-SAE experiment suite.
# Inherit pytorch+CUDA from official image; bake deps + repo, defer config to runtime.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# HF_HOME points into the mounted cache directory on purpose. COCO is loaded
# without streaming and its download is about 20 GB; left in the container's
# writable layer that needs the space free on the storage driver, and
# `docker run --rm` discards it, so an extraction that is interrupted and
# restarted re-downloads all of it even though the chunk writer itself resumes.
# Under /app/repo/cache the downloads outlive the container and sit on the same
# volume as the embeddings they produce.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/app/repo/cache/hf \
    PYTHONPATH=/app/repo

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/repo
# .dockerignore keeps the local .venv, cache/ and outputs/ out of the build
# context; the last two are bind-mounted at run time instead.
COPY . /app/repo
RUN pip install --no-cache-dir -e /app/repo

# Mount cache + outputs at runtime, e.g.:
#   docker run --rm --gpus all \
#     -e HF_TOKEN=$HF_TOKEN \
#     -e CONFIG=configs/post_rebuttal/clip_b32_cc3m.yaml \
#     -v $PWD/cache:/app/repo/cache -v $PWD/outputs:/app/repo/outputs \
#     vlm-sae
#
# Size the mounted cache volume for both halves of what lands there: the
# Hugging Face downloads under cache/hf (COCO alone is about 20 GB) and the
# embedding caches beside them.

RUN cp /app/repo/docker/entrypoint.sh /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
