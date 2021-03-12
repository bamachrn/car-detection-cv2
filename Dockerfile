FROM python:3.8-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir -p /opt/model_data/
COPY model_data /opt/model_data/
RUN mkdir -p /app
COPY car_detection /app
WORKDIR /app
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
