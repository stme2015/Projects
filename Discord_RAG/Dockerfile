# Use Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose ports
EXPOSE 8000 8001

# Start FastAPI app
CMD ["uvicorn", "app.main:fastapp", "--host", "0.0.0.0", "--port", "8000"]
