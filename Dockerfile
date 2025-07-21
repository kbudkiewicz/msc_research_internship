#FROM python:3.10
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
LABEL description='Image for MSc research internship (SoSe 2025). Flow-Matching for Generative Medical Imaging'

# make sure deployment path point to the same repo as below
#VOLUME /home/kbudkiewicz/msc_internship /msc_internship

WORKDIR /msc_internship
RUN apt update
RUN apt -y install tmux

COPY ./requirements.txt .
RUN pip install -r requirements.txt
