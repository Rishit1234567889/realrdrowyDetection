# Solution 1: Use conda to install pre-compiled dlib (RECOMMENDED)
FROM continuumio/miniconda3:latest

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create conda environment
RUN conda create -n drowsiness python=3.9 -y

# Activate environment and install packages
RUN echo "source activate drowsiness" > ~/.bashrc
ENV PATH /opt/conda/envs/drowsiness/bin:$PATH

# Install dlib and opencv via conda (pre-compiled)
RUN conda install -n drowsiness -c conda-forge dlib opencv -y


# Create directory for cascade files
RUN mkdir -p /app/data/haarcascades

# Download Haar cascade files
RUN wget -O /app/data/haarcascades/haarcascade_frontalface_default.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

RUN wget -O /app/data/haarcascades/haarcascade_eye.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

# Install remaining packages via pip
COPY requirements_conda.txt ./requirements.txt
RUN /opt/conda/envs/drowsiness/bin/pip install -r requirements.txt

COPY . .

CMD ["/opt/conda/envs/drowsiness/bin/python", "app.py"]

# =============================================================
# Solution 2: Use Ubuntu with pre-compiled dlib
# =============================================================
# FROM ubuntu:20.04
# 
# ENV DEBIAN_FRONTEND=noninteractive
# 
# # Install Python and dlib from system packages
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev \
#     python3-dlib \
#     python3-opencv \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# 
# # Install remaining Python packages
# COPY requirements_ubuntu.txt ./requirements.txt
# RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install -r requirements.txt
# 
# COPY . .
# 
# CMD ["python3", "main.py"]

# =============================================================
# Solution 3: Use pre-built dlib wheel
# =============================================================
# FROM python:3.9-slim
# 
# # Install minimal runtime dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     libgtk2.0-dev \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# 
# COPY requirements.txt .
# 
# # Try to install dlib from pre-built wheel
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir \
#     --find-links https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases/download/v19.22.99/dlib-19.22.99-cp39-cp39-linux_x86_64.whl \
#     dlib || pip install --no-cache-dir dlib
# 
# RUN pip install --no-cache-dir -r requirements.txt
# 
# COPY . .
# 
# CMD ["python", "main.py"]

# =============================================================
# Solution 4: Multi-stage build (if you must compile)
# =============================================================
# # Build stage
# FROM python:3.9 as builder
# 
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     libopenblas-dev \
#     liblapack-dev \
#     libx11-dev \
#     libgtk-3-dev \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*
# 
# # Install dlib in builder stage
# RUN pip install --user dlib==19.24.2
# 
# # Runtime stage
# FROM python:3.9-slim
# 
# # Copy compiled dlib from builder
# COPY --from=builder /root/.local /root/.local
# 
# # Install runtime dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# 
# # Add local packages to path
# ENV PATH=/root/.local/bin:$PATH
# ENV PYTHONPATH=/root/.local/lib/python3.9/site-packages:$PYTHONPATH
# 
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# 
# COPY . .
# 
# CMD ["python", "main.py"]