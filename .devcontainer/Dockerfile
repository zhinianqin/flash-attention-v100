FROM mcr.microsoft.com/azureml/inference-base-cuda12.4-ubuntu22.04:20260218.v1
RUN /opt/miniconda/bin/conda install git -y --no-update-deps && \
/opt/miniconda/bin/conda clean -y --all && \
/opt/miniconda/bin/pip install --no-cache-dir "cmake>=3.26" ninja packaging "setuptools>=49.4.0" wheel jinja2 && \
/opt/miniconda/bin/pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124