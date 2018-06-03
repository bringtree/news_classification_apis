FROM tensorflow/tensorflow:latest-py3
MAINTAINER Bringtree "bringtree@qq.com"
RUN apt-get update && apt-get -y install git
RUN pip3 install Flask
RUN pip3 install jieba
RUN pip3 install joblib
RUN pip3 install gunicorn gevent

WORKDIR /home/
RUN git clone https://github.com/bringtree/news_classification_apis.git
WORKDIR  /home/project_design_app/server
CMD ["python3", "app:app", "-c", "./gunicorn.conf.py"]

