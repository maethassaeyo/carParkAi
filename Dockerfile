FROM python:3.9-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" "opencv-python-headless<5" flask
COPY . /app

EXPOSE 5000



CMD ["python","demo.py"]
