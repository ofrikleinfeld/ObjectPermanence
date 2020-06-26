# pull nvidia cuda 10 ubuntu 18 base image
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# update apt and install wget miniconda
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# copy project files to container and chage working directory
ENV PROJECT_DIR="/ObjectPermanence"
COPY . $PROJECT_DIR
WORKDIR $PROJECT_DIR

# create a conda environment with the required packages and activate it
RUN conda env update --name base -f environment.yml

ENTRYPOINT /bin/bash
