FROM python:3.9-slim-buster
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0
WORKDIR test
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .
COPY assets assets


