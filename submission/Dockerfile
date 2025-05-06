FROM python:3.13

RUN pip3 install click tira

ADD style-pipeline.py /

ENTRYPOINT [ "/style-pipeline.py" ]