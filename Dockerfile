FROM dolfinx/dolfinx:stable
LABEL maintainer="Pierric Mora <pierric.mora@univ-eiffel.fr>"
LABEL description=" elastodynamics with Fenicsx/dolfinx "

# Install tkinter for matplotlib plots
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
RUN apt-get update && \
    apt-get install -y python3-tk && \
    apt-get clean

WORKDIR /tmp

# Install elastodynamicsx
ADD . /tmp/

RUN pip3 install .

RUN rm -rf /tmp/*

# Install optional dependencies
RUN pip3 install tqdm

WORKDIR /root
