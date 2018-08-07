# ML Inference Training

## Install

Training files assume Ubuntu 16.04.

* Install Miniconda3:
```
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash -c Miniconda3-latest-Linux-x86_64.sh
    export PATH=$HOME/miniconda3/bin/conda:$PATH
```

* Setup environment:

Install on a GPU equipped machine.  See [here](https://github.com/lukmanr/gcp-gpu-install/tree/master/ubuntu) for
installing CUDA drivers.

    conda create -n mli
    source activate mli
    conda install --file conda.txt
    pip install -r requirements.txt
    pip install tensorflow-gpu


* Build TensorFlow tools
```
    sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip
    sudo apt-get install -y python3-numpy python3-dev python3-pip python3-wheel
    wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh
    ./bazel-0.16.0-installer-linux-x86_64.sh --user
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    git checkout r1.10
    bazel build tensorflow/tools/graph_transforms:transform_graph
    bazel build tensorflow/tools/graph_transforms:summarize_graph
```