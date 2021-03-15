FROM python:3.8-buster
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/

Run apt update 
RUN apt -y -qq install python3-opencv 

RUN pip install -r requirements.txt
