FROM python

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi


RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]