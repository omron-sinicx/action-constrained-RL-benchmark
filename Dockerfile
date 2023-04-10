FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update \
 && apt-get install -y \ 
    git \
    swig \
    tmux


#ENV DEBIAN_FRONTEND=noninteractive
#RUN apt update && apt install -y xvfb x11vnc python-opengl icewm
#RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> /root/.bashrc
#
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    zip \
    xserver-xorg-dev \
    cython \ 
    libglpk-dev \
    task-spooler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ARG USERNAME=miura
ARG USER_UID=1011
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

#RUN mv /root/.mujoco /home/${USERNAME}
RUN mkdir -p /home/${USERNAME}/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip \
    && mv mjpro150 /home/${USERNAME}/.mujoco \
    && rm mujoco.zip

RUN wget https://www.roboti.us/file/mjkey.txt \
    && mv mjkey.txt /home/${USERNAME}/.mujoco

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV LD_LIBRARY_PATH=/home/${USERNAME}/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}:
ENV MUJOCO_PY_MUJOCO_PATH=/home/${USERNAME}/.mujoco/mjpro150
COPY ./requirements.txt /home/${USERNAME}
RUN pip install -r /home/${USERNAME}/requirements.txt
#RUN chmod -R 777 /root/.mujoco

# install PyBullet-Gym
RUN cd /home/${USERNAME}/ && git clone https://github.com/benelot/pybullet-gym.git \
    && cd pybullet-gym \
    && pip install -e . \
    && cd .. && cd