# ML Inference Training

## Install

Assumes debian Linux.

* Install Miniconda3:
```
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod a+x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    export PATH=$HOME/miniconda3/bin:$PATH
```

* Setup environment:

Install on a GPU equipped machine.  See [here](https://github.com/lukmanr/gcp-gpu-install/tree/master/ubuntu) for
installing CUDA drivers if necessary.

    conda create -y -n mli
    source activate mli
    conda install -y --file conda.txt
    pip install -r requirements.txt
    pip install tensorflow-gpu

* Prep for TensorFlow build

```
    sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip
    sudo apt-get install -y python3-numpy python3-dev python3-pip python3-wheel
    wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh
    chmod a+x bazel-0.16.0-installer-linux-x86_64.sh
    ./bazel-0.16.0-installer-linux-x86_64.sh --user
    export PATH=$HOME/bin:$PATH
```

* Build TensorFlow tools

```
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    git checkout r1.9
    bazel build tensorflow/tools/graph_transforms:transform_graph
    bazel build tensorflow/tools/graph_transforms:summarize_graph
```