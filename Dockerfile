# Use the official Python image as a base image
FROM python:3.10-slim

WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY checkpoints/model_checkpoint.pth checkpoints/model_checkpoint.pth
COPY data/test data/test

COPY src/ src/

RUN pwd && ls -la
CMD ["python", "src/test.py"]
