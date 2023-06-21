FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update

COPY requirements.txt /opt/movies/requirements.txt
COPY dataset/dataset_True_True_True_True_True.csv /opt/movies/dataset/dataset_True_True_True_True_True.csv
COPY train.py /opt/movies/train.py

WORKDIR /opt/movies
RUN python3 -m venv venv
RUN pip install -r requirements.txt


# commands to build: docker build -t movies-nlp-img .
# commands to start: docker run -it --name movies-nlp movies-nlp-img
# enter with: docker exec -it movies-nlp-img /bin/bash
# then run: cd /opt/movies && python3 train.py
