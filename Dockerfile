FROM tensorflow/tensorflow:latest-py3
MAINTAINER Bringtree "bringtree@qq.com"

RUN sed -i 's#http://archive.ubuntu.com#http://mirrors.163.com#g' /etc/apt/sources.list
RUN apt-get update && apt-get -y install git

RUN pip3 install Flask
RUN pip3 install jieba
RUN pip3 install joblib
RUN pip3 install gunicorn gevent
RUN mkdir /root/.pip
RUN echo "[global]" > /root/.pip/pip.conf
RUN echo "index-url = https://mirrors.scau.edu.cn/pypi/web/simple" >> /root/.pip/pip.conf
WORKDIR /home/
RUN git clone https://github.com/bringtree/news_classification_apis.git

EXPOSE 5000
WORKDIR /home/news_classification_apis/server/
CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]
