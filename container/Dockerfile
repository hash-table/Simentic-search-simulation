FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    apt-utils \
	build-essential \
    locales \
    tzdata \
    libssl-dev \
    curl \
    wget \
    unzip \
    vim \
    git \
    htop \
    net-tools \
    default-jdk


RUN apt-get install -y python-dev && apt-get update

RUN apt-get install -y \
	libmysqlclient-dev \
	python-opencv \
    python-numpy \
    python-scipy && \
    rm -rf /var/lib/apt/lists/* \
    rm -rf /var/cache/apk/*w

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8
ENV TZ=Asia/Seoul

RUN pip install transformers \
    pip install -U sentence-transformers

ARG PROJECT_HOME=/usr/src/my-sbert-search-app/

RUN pwd
COPY ./data ${PROJECT_HOME}/data
COPY ./embedding ${PROJECT_HOME}/embedding
COPY ./result ${PROJECT_HOME}/result
COPY ./util ${PROJECT_HOME}/util
COPY ./init_data_pool.py ${PROJECT_HOME}/init_data_pool.py
COPY ./main.py ${PROJECT_HOME}/main.py

WORKDIR  ${PROJECT_HOME}