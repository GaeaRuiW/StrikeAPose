FROM dustynv/pytorch:2.1-r35.4.1

WORKDIR /app

RUN pip3 install -U matplotlib pillow pyyaml requests pandas py-cpuinfo scipy tqdm ultralytics-thop
RUN pip3 install ultralytics --no-deps
RUN pip3 install torchvision==0.16 --no-deps

COPY ./ffmpeg.tgz /app/
RUN tar zxvf ffmpeg.tgz

RUN git clone https://github.com/Keylost/jetson-ffmpeg.git && \
    cd jetson-ffmpeg && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j6 && \
    make install && \
    ldconfig && \
    cd .. && \
    ./ffpatch.sh ../ffmpeg && \
    cd ../ffmpeg && \
    ./configure --enable-nvmpi && \
    make -j6 && \
    make install && \
    cd /app

RUN rm -rf jetson-ffmpeg && rm -rf ffmpeg*
