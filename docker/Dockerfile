FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y git libgl1-mesa-dev libopencv-dev python3-dev python3-pip python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 install \
    openmim fastapi uvicorn[standard] python-multipart
RUN mim install -y mmcv-full \
    && mim install -y mmdet \
    && mim install -y mmpose

RUN pip3 install git+https://github.com/hysts/anime-face-detector

COPY src /face_pose_detector
WORKDIR /face_pose_detector