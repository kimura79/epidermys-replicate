FROM r8.im/cog/python:3.10

RUN apt-get update && apt-get install -y libgl1

COPY . /src
WORKDIR /src

RUN pip install -r requirements.txt
