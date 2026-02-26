FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt-get update && \
apt-get install -y --no-install-recommends curl git ssh && \
rm -rf /var/lib/apt/lists/* && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
export UV_BREAK_SYSTEM_PACKAGES=1 && \
/root/.local/bin/uv python install 3.9 && \
/root/.local/bin/uv pip install --no-cache --system "cmake>=3.26" ninja packaging "setuptools>=49.4.0" wheel jinja2 && \
/root/.local/bin/uv pip install --no-cache --system torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
