#FROM python:2.7
FROM sagemath/sagemath
MAINTAINER Andriy Mulyar <andriy.mulyar@gmail.com>

USER root
RUN apt update && sudo apt upgrade -y zlib1g && apt install -y python-pip libpq-dev && pip install --upgrade pip
# Create the group and user to be used in this container
#RUN groupadd flaskgroup && useradd -m -g flaskgroup -s /bin/bash flask && usermod -a -G sage flask && usermod -a -G flaskgroup sage


# Create the working directory (and set it as the working directory)
RUN mkdir -p /home/flask/app/api
WORKDIR /home/flask/app/api

# Install the package dependencies (this step is separated
# from copying all the source code to avoid having to
# re-install all python packages defined in requirements.txt
# whenever any source code change is made)
COPY requirements.txt /home/flask/app/api

#RUN pip install Flask

#ln -sf /home/sage/sage/local/bin/flask /usr/local/bin/flask

#RUN pip uninstall --yes flask && pip install --yes flask

# Copy the source code into the container
COPY . /home/flask/app/api
RUN chown -R sage:sage /home/flask
ENV PYTHONUSERBASE /home/sage/sage/local

USER sage
RUN pip install --user -r requirements.txt

USER root
RUN mv /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1.old
RUN ln -s /home/sage/sage/local/lib/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1
USER sage
