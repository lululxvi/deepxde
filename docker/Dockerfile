FROM horovod/horovod:latest
LABEL description="Deepxde with Horovod and GPU support"

COPY requirements.txt /root
WORKDIR /root
RUN pip3 install -r requirements.txt 
RUN apt-get update
RUN apt-get install -y imagemagick
EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
