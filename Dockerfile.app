FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install llama-cpp-python
RUN pip install faiss-cpu
# Copy application code
COPY . .

# Expose Flask port
EXPOSE 8070

# Run the Flask app
CMD ["python", "main_app.py"]
