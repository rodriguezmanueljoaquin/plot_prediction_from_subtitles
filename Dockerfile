FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update

COPY requirements.txt /opt/movies/requirements.txt

WORKDIR /opt/movies
RUN python3 -m venv venv
RUN pip install -r requirements.txt

ENTRYPOINT ["sleep", "infinity"]

# commands to build: docker build -t movies-nlp-img .
# commands to start: docker run -it --name movies-nlp movies-nlp-img
# enter with: docker exec -it movies-nlp-img /bin/bash
# then run: cd /opt/movies && python3 train.py
