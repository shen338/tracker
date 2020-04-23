FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# author
MAINTAINER Tong Shen

# extra metadata
LABEL description="Tracker environment, include pysot, detectron2, deep-person-reid"

RUN apt update 

RUN apt install git
RUN apt install unzip

RUN git clone https://github.com/shen338/tracker

RUN cd tracker

RUN unzip fake_dataset.zip

# install pytorch
RUN apt install python3
RUN apt install -y python3-distutils
RUN apt install -y curl 
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

RUN pip install torch torchvision


# build tracker
RUN cd tracking 

RUN pip install opencv-python
RUN pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
RUN apt-get install python3-dev
RUN python3 setup.py build_ext --inplace

# install detectron2 
RUN cd ../detectron2

RUN pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN python3 -m pip install -e .

# install reid module
RUN cd ../deep-person-reid/

RUN pip install -r requirements.txt
RUN apt-get install -y libgtk2.0-dev
RUN python3 setup.py develop

RUN cd ..

# install filterpy
RUN pip install filterpy