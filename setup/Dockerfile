FROM pytorch/pytorch:latest

RUN apt-get update -y 
RUN apt-get upgrade -y

RUN apt-get install --no-install-recommends -y \
  wget git swig ffmpeg \
	nodejs sudo vim 

RUN pip install --upgrade pip

RUN pip install --upgrade scipy
RUN pip install --upgrade sklearn
RUN pip install --upgrade matplotlib

RUN pip install --upgrade tqdm

RUN pip install --upgrade tensorboard

# Install Jupyter notebook related packages
RUN pip install --upgrade jupyterlab
RUN pip install --upgrade notebook
RUN pip install --upgrade voila

# Install Graph ML related packages
RUN pip install --upgrade networkx

# Workaround to be able to install torch-scatter

RUN pip install --upgrade -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --upgrade -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --upgrade -q torch-geometric
RUN pip install ogb
RUN pip install -q git+https://github.com/snap-stanford/deepsnap.git 

ENV HOME /home/shurubaru
RUN mkdir -p ${HOME}

WORKDIR /home/shurubaru
