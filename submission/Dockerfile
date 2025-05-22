FROM python:3.9

# set up env
ADD requirements.txt /
RUN pip3 install click tira
RUN pip3 install -r requirements.txt

# add files
ADD models.py /
ADD style-pipeline.py /


ENTRYPOINT [ "/style-pipeline.py" ]