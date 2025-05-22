FROM python:3.9

# set up env
ADD requirements.txt /
RUN pip3 install click tira
RUN pip3 install -r requirements.txt

# add files
ADD models.py /
ADD style-pipeline.py /
ADD all-MiniLM-L12-v2-Best.pth /
ADD all-mpnet-base-v2-Best.pth /
ADD deberta-base-Best.pth /
ADD roberta-base-Best.pth /
ADD sentence-t5-base-Best.pth /

ENTRYPOINT [ "/style-pipeline.py" ]