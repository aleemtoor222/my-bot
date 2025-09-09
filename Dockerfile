FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
 
# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
 
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libsqlite3-dev \
    libgomp1 \
    libstdc++6 \
    ffmpeg \
&& rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /app
 
# CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 
# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt
 
# Install llama-cpp-python with GPU support
RUN pip install --no-cache-dir --force-reinstall --upgrade llama-cpp-python
 
# Copy application files
COPY . .
 
# Expose port
EXPOSE 8070
 
# Run the app
CMD ["python", "main_app.py"]