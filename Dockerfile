FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure uvicorn is available
RUN pip install uvicorn[standard]

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "backend_integrated_updated_complete:app", "--host", "0.0.0.0", "--port", "8000"]
