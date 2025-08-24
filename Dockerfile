FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv
WORKDIR /workspace

# 1) System deps + venv support (+ OpenCV runtime deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libgomp1 \
    libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2) Create & activate a dedicated virtualenv (avoids PEP 668)
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3) Modern pip inside the venv
RUN pip install --upgrade pip

# 4) PyTorch + CUDA 12.6 wheels (Python 3.12-compatible)
#    Pairing: torch 2.6.0 ↔ torchvision 0.21.0
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0

# 5) Common libs (headless OpenCV + ultralytics dependencies)
RUN pip install --no-cache-dir opencv-python-headless pillow numpy matplotlib pyyaml tqdm \
    requests psutil pandas seaborn ultralytics dill pywavelets timm

# 6) Your code
COPY . .

CMD ["/bin/bash"]
