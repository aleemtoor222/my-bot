# Base Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies (only required ones)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1



# Install remaining requirements (make sure torch, torchvision, torchaudio are REMOVED from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose FastAPI/Flask port
EXPOSE 8070

# Start app with uvicorn (for FastAPI)
CMD ["python3", "main_app.py"]  
