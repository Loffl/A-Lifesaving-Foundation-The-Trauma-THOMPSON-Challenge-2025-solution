FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS t3-challenge-actionclip

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/opt/app:$PYTHONPATH
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install system dependencies as root before switching to user
#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Install additional PyTorch packages with CUDA support first (to match your local setup) --> now in req. setup
#RUN python -m pip install \
#    --user \
#    --no-cache-dir \
#    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
#    -f https://download.pytorch.org/whl/torch_stable.html \
#    --timeout 600 --retries 10

# Then install other requirements
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]