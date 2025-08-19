FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (rarely change - cache efficiently)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (cache layer)
RUN pip install --upgrade pip

# Install PyTorch ecosystem dependencies first (already in base image but ensure compatible versions)
RUN pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies with compatible versions
RUN pip install \
    pillow \
    tqdm \
    pyyaml \
    matplotlib \
    seaborn \
    pandas \
    tensorboard

# Install computer vision and Ultralytics (will pull correct NumPy)
RUN pip install \
    "opencv-python-headless>=4.8.0" \
    "ultralytics>=8.0.196"

# Reinstall NumPy to fix any version conflicts (Ultralytics best practice)
RUN pip install "numpy>=1.23.5,<2.0"

# Install additional ML tools
RUN pip install \
    thop

# Install UAV-DETR specific dependencies (most likely to change)
RUN pip install \
    dill \
    PyWavelets \
    timm \
    gdown

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/runs/detect /workspace/evaluation_results

# Set permissions
RUN chmod +x /workspace/train_halo.py /workspace/evaluate_halo.py

# Default command
CMD ["/bin/bash"]
