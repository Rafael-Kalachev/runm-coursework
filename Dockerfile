FROM python:3.8-slim-buster

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get -y install gcc

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy all files in the current dir to the main dir of the container
#COPY . .
CMD [ "python3", "-W ignore" ,"training.py" ]

