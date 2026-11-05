ARG BASE_IMAGE=fastsem/nitroslam:jetson-r36.3-tumvi-speed-nostats
FROM ${BASE_IMAGE}

ARG BUILD_JOBS=2

COPY scripts/patch_nitro_with_semantics.py /usr/local/bin/patch_nitro_with_semantics.py
COPY semantic /tmp/dynslam_semantic

WORKDIR /opt/nitroslam
RUN python3 /usr/local/bin/patch_nitro_with_semantics.py /opt/nitroslam /tmp/dynslam_semantic \
    && cmake -S /opt/nitroslam -B /opt/nitroslam/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/nitroslam/build --target rgbd_tum -j"${BUILD_JOBS}"

ENV DYNSLAM_INTEGRATED=1
