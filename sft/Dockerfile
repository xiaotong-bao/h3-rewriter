FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/trl_sft_official_v2_20261006
