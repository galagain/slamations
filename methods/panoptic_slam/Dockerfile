FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

ENV DEBIAN_FRONTEND=noninteractive \
    PANOPTIC_SLAM_ROOT=/opt/panoptic_slam

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https build-essential ca-certificates checkinstall cmake curl \
    gfortran git libatlas-base-dev libavcodec-dev libavformat-dev libavutil-dev \
    libboost-all-dev libeigen3-dev libegl1-mesa-dev libglew-dev libgl1-mesa-dev \
    libgoogle-glog-dev libgflags-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev libgtk-3-dev libgtk2.0-dev libjpeg-dev libleveldb-dev \
    liblmdb-dev libpng-dev libprotobuf-dev libssl-dev libswscale-dev libtbb-dev \
    libtiff-dev libv4l-dev libx264-dev libxvidcore-dev ninja-build pkg-config \
    protobuf-compiler python3-dev python3-numpy python3-opencv python3-pip \
    python3-venv unzip wget zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/catchorg/Catch2.git \
 && cmake -S Catch2 -B Catch2/build -DBUILD_TESTING=OFF \
 && cmake --build Catch2/build --target install -j"$(nproc)"

RUN git clone --branch v0.5 --depth 1 https://github.com/stevenlovegrove/Pangolin.git Pangolin-v0.5 \
 && find Pangolin-v0.5 -name CMakeLists.txt -exec sed -i -E 's/-Werror(=[^ ")]*)?//g' {} + \
 && cmake -S Pangolin-v0.5 -B Pangolin-v0.5/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PANGOLIN_VIDEO=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DCMAKE_C_FLAGS="-Wno-error" \
      -DCMAKE_CXX_FLAGS="-Wno-error -Wno-error=maybe-uninitialized -Wno-error=catch-value -Wno-error=deprecated-declarations" \
 && find Pangolin-v0.5/build -name flags.make -exec sed -i -E 's/(^| )-Werror[^ ]*//g' {} + \
 && cmake --build Pangolin-v0.5/build -j"$(nproc)" \
 && cmake --install Pangolin-v0.5/build \
 && ldconfig

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY wheels /opt/panoptic_wheels
COPY src ${PANOPTIC_SLAM_ROOT}

ARG INSTALL_DETECTRON2=1
WORKDIR ${PANOPTIC_SLAM_ROOT}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev libopenblas0 \
 && rm -rf /var/lib/apt/lists/*
RUN chmod +x /usr/local/bin/*.sh /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && python3 -m venv .venv \
 && . .venv/bin/activate \
 && pip install --upgrade pip wheel setuptools \
	 && pip install 'numpy==1.23.5' opencv-python pyyaml yacs tqdm cython easydict 'pillow<10' \
	 && if [ "$INSTALL_DETECTRON2" = "1" ]; then \
	      if ls /opt/panoptic_wheels/torch-*.whl /opt/panoptic_wheels/torchvision-*.whl >/dev/null 2>&1; then \
	        pip install /opt/panoptic_wheels/torch-*.whl /opt/panoptic_wheels/torchvision-*.whl; \
	      else \
	        pip install 'torch<2.4' 'torchvision<0.19'; \
	      fi && \
	      pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@v0.5'; \
	    fi \
 && if [ -f requirements.txt ]; then pip install -r requirements.txt || true; fi \
 && pip install --force-reinstall 'numpy==1.23.5'

# The upstream CMake file is pinned to x86_64, Python 3.8 and CUDA 11.6.
# Patch it for JetPack 6 / aarch64 before building.
RUN python3 - <<'PY'
from pathlib import Path
p = Path("CMakeLists.txt")
s = p.read_text()
s = s.replace("set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)", "set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)")
s = s.replace("set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy)", "execute_process(COMMAND python3 -c \"import numpy; print(numpy.get_include())\" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)")
s = s.replace("/usr/include/python3.8/\n/usr/lib/python3.8/dist-packages/numpy/core/include/numpy/", "${PYTHON_INCLUDE_DIRS}\n${NUMPY_INCLUDE_DIR}")
s = s.replace("/usr/lib/x86_64-linux-gnu/libpython3.8.so", "${PYTHON_LIBRARIES}")
p.write_text(s)

p = Path("src/python/panoptic_python.py")
s = p.read_text()
s = s.replace("import argparse\n", "import argparse\nimport os\nimport torch\n")
s = s.replace(
    "        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold\n"
    "        cfg.freeze()\n",
    "        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold\n"
    "        cfg.MODEL.DEVICE = os.environ.get(\"PANOPTIC_SLAM_DEVICE\", \"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
    "        cfg.freeze()\n",
)
p.write_text(s)

s = p.read_text()
s = s.replace(
    """        #fill results with "Things" 
        #[id,isThing,score,category_id,instance_id,area,mask,bbox]
        num_instances = len(instance_mask)
        semantic_labels = [] # store semantic label 

        for info,pred_mask,boxes in zip(panoptic_result,instance_mask,bboxes):
            tmp=[]
            
            info_items =list(info.items()) # get each item from dict panoptic_result and convert into a list

            pred_mask = np.array(pred_mask*255).astype('uint8')         

            #if isThing is False
            if(info_items[1][1] is False): continue # ignore Stuff objects 
            
            tmp.append(info_items[0][1]) # id
            tmp.append(info_items[1][1]) # isThing
            tmp.append(info_items[2][1]) #score
            tmp.append(info_items[3][1]) #category_id
            tmp.append(info_items[4][1]) #instance_id
            tmp.append(info_items[5][1]) #area
            tmp.append(self.binary_mask_2_bytearray(pred_mask)) # numpy mask
            tmp.append(list(boxes)) # numpy bbox
            results.append(tmp)
            self.all_masks.append(pred_mask)
     
        #fill results with "stuff" 
        semantic_results = panoptic_result[num_instances:]
        
        for info in semantic_results:
            info_items =list(info.items())
            semantic_labels.append(info_items[2][1])
   
        for x in np.unique(semantic_mask):
            if(x==0): 
                binary_mask = np.array(semantic_mask)==x
                self.union_instance_mask = binary_mask
            else:
                if(x in semantic_labels):
                    idx = semantic_labels.index(x)
                    binary_mask = np.array(semantic_mask)==x
                    semantic_result = semantic_results[idx]

                    info_items =list(semantic_result.items())
                    tmp = []

                    binary_mask = np.array(binary_mask*255).astype('uint8')

                    tmp.append(info_items[0][1]) # id
                    tmp.append(info_items[1][1]) # isThing
                    tmp.append(0.0) #score
                    tmp.append(info_items[2][1]) #category_id
                    tmp.append(0) #instance_id
                    tmp.append(info_items[3][1]) #area
                    tmp.append(self.binary_mask_2_bytearray(binary_mask)) # numpy mask
                    tmp.append([0,0,1,1]) # numpy bbox
                    results.append(tmp)
                    self.all_masks.append(binary_mask)
""",
    """        #fill results with "Things" 
        #[id,isThing,score,category_id,instance_id,area,mask,bbox]
        thing_results = [info for info in panoptic_result if info.get('isthing', info.get('isThing', False))]
        semantic_results = [info for info in panoptic_result if not info.get('isthing', info.get('isThing', False))]
        semantic_by_label = {info.get('category_id', -1): info for info in semantic_results}
        self.union_instance_mask = np.zeros(semantic_mask.shape, dtype=bool)

        for info,pred_mask,boxes in zip(thing_results,instance_mask,bboxes):
            tmp=[]
            pred_mask = np.array(pred_mask*255).astype('uint8')
            tmp.append(info.get('id', 0)) # id
            tmp.append(bool(info.get('isthing', info.get('isThing', True)))) # isThing
            tmp.append(float(info.get('score', 0.0))) #score
            tmp.append(int(info.get('category_id', 0))) #category_id
            tmp.append(int(info.get('instance_id', info.get('id', 0)))) #instance_id
            tmp.append(int(info.get('area', int(np.count_nonzero(pred_mask))))) #area
            tmp.append(self.binary_mask_2_bytearray(pred_mask)) # numpy mask
            tmp.append(list(boxes)) # numpy bbox
            results.append(tmp)
            self.all_masks.append(pred_mask)
     
        #fill results with "stuff" 
        for x in np.unique(semantic_mask):
            if(x==0): 
                binary_mask = np.array(semantic_mask)==x
                self.union_instance_mask = binary_mask
            elif(int(x) in semantic_by_label):
                binary_mask = np.array(semantic_mask)==x
                semantic_result = semantic_by_label[int(x)]
                tmp = []

                binary_mask = np.array(binary_mask*255).astype('uint8')

                tmp.append(semantic_result.get('id', int(x))) # id
                tmp.append(False) # isThing
                tmp.append(0.0) #score
                tmp.append(int(semantic_result.get('category_id', int(x)))) #category_id
                tmp.append(0) #instance_id
                tmp.append(int(semantic_result.get('area', int(np.count_nonzero(binary_mask))))) #area
                tmp.append(self.binary_mask_2_bytearray(binary_mask)) # numpy mask
                tmp.append([0,0,1,1]) # numpy bbox
                results.append(tmp)
                self.all_masks.append(binary_mask)
""",
)
p.write_text(s)

p = Path("src/python/predictor.py")
s = p.read_text()
s = s.replace("import cv2\nimport torch\n", "import cv2\nimport os\nimport torch\n")
s = s.replace(
    "        predictions = self.predictor(image)\n\n"
    "        # Convert image from OpenCV BGR format to Matplotlib RGB format.\n",
    "        predictions = self.predictor(image)\n"
    "        if os.environ.get(\"PANOPTIC_SLAM_SKIP_VISUALIZER\", \"1\") == \"1\":\n"
    "            class _RawVis:\n"
    "                def __init__(self, bgr):\n"
    "                    self._image = bgr[:, :, ::-1].copy()\n"
    "                def get_image(self):\n"
    "                    return self._image\n"
    "            return predictions, _RawVis(image)\n\n"
    "        # Convert image from OpenCV BGR format to Matplotlib RGB format.\n",
)
p.write_text(s)

p = Path("Examples/RGB-D/rgbd_tum.cc")
s = p.read_text()
s = s.replace(
    "    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);",
    "    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,false);",
)
s = s.replace(
    "    // Stop all threads\n"
    "    SLAM.Shutdown();\n",
    "    // Save before shutdown so batch runs keep a trajectory even if cleanup stalls.\n"
    "    SLAM.SaveTrajectoryTUM(\"./Output/CameraTrajectory.txt\");\n"
    "    SLAM.SaveKeyFrameTrajectoryTUM(\"./Output/KeyFrameTrajectory.txt\");\n"
    "    SLAM.SaveTrajectoryTUM(\"CameraTrajectory.txt\");\n"
    "    SLAM.SaveKeyFrameTrajectoryTUM(\"KeyFrameTrajectory.txt\");\n\n"
    "    // Stop all threads\n"
    "    SLAM.Shutdown();\n",
)
p.write_text(s)
PY

RUN . .venv/bin/activate \
 && if [ -f Vocabulary/ORBvoc.txt.tar.gz ] && [ ! -f Vocabulary/ORBvoc.txt ]; then tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary; fi \
 && bash build.sh

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
