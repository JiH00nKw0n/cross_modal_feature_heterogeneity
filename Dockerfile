# GPU image for the VLM-SAE experiment suite.
# Inherit pytorch+CUDA from official image; bake deps + repo, defer config to runtime.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/app/repo

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/repo
COPY . /app/repo
RUN pip install --no-cache-dir -e /app/repo

# Mount cache + outputs at runtime, e.g.:
#   docker run --rm --gpus all \
#     -e HF_TOKEN=$HF_TOKEN \
#     -e CONFIG=configs/cc3m/overrides/clip_l14.yaml \
#     -v $PWD/cache:/app/repo/cache -v $PWD/outputs:/app/repo/outputs \
#     vlm-sae

RUN cp /app/repo/docker/entrypoint.sh /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
