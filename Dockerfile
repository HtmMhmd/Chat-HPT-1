# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 5000 available for the MLflow UI
EXPOSE 5000

# Create volume mount points
VOLUME ["/app/data", "/app/output", "/app/mlruns"]

# Set environment variables
ENV PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=/app/mlruns

# Run app when the container launches
ENTRYPOINT ["python", "run.py"]
