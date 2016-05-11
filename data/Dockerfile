FROM ubuntu:14.04
MAINTAINER Noushin Niknafs <niknafs@jhu.edu>

# install required packages
RUN apt-get update
RUN apt-get -y install python
RUN apt-get -y install git
RUN apt-get -y install python-pip
RUN apt-get -y install python-dev

RUN apt-get install build-essential
RUN apt-get -y install libcairo2-dev
RUN apt-get install -y python-cairo
RUN apt-get install wget

RUN apt-get install -y python-numpy
RUN apt-get install -y python-scipy
RUN apt-get install -y python-matplotlib
RUN apt-get install -y python-sklearn
RUN apt-get install -y python-yaml
RUN apt-get install -y python-setuptools

RUN apt-get install -y libglpk-dev  
RUN apt-get install -y  libgmp3-dev
RUN apt-get install -y  libblas-dev
RUN apt-get install -y  liblapack-dev
RUN apt-get install -y  libarpack2-dev


RUN apt-get install -y  libxml2-dev


RUN wget http://igraph.org/nightly/get/c/igraph-0.7.1.tar.gz
RUN tar zxvf igraph-0.7.1.tar.gz

WORKDIR igraph-0.7.1
RUN ./configure
RUN make
RUN make install
WORKDIR ..

RUN export LD_LIBRARY_PATH=/usr/local/lib/
ENV LD_LIBRARY_PATH /usr/local/lib/

RUN wget https://pypi.python.org/packages/source/p/python-igraph/python-igraph-0.7.1.post6.tar.gz
RUN tar zxvf python-igraph-0.7.1.post6.tar.gz

WORKDIR python-igraph-0.7.1.post6
RUN python setup.py build
RUN python setup.py install
WORKDIR ..

# add our python app code to the image
RUN git clone https://github.com/KarchinLab/SCHISM.git
WORKDIR /SCHISM/
RUN cat setup.py 
RUN python setup.py sdist
RUN pip install dist/SCHISM-1.1.1.tar.gz

ENV LD_LIBRARY_PATH /usr/local/lib/

WORKDIR ..
