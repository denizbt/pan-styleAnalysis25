FROM --platform=linux/amd64 python:3.9-slim

# set up env
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install click tira
RUN pip3 install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0+cu118
RUN pip3 install -r requirements.txt

# add files
ADD models.py /
ADD style-pipeline.py /
ADD all-MiniLM-L12-v2-Best.pth /
ADD all-mpnet-base-v2-Best.pth /
ADD deberta-base-Best.pth /
ADD roberta-base-Best.pth /
ADD sentence-t5-base-Best.pth /

# ensure script is executable
RUN chmod +x /style-pipeline.py
ENTRYPOINT [ "/style-pipeline.py" ]