# tensorrt-unet
This is a TensorRT version Unet, inspired by [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [pytorch-unet](https://github.com/milesial/Pytorch-UNet).<br>
You can generate TensorRT engine file using this script and customize some params and network structure based on network you trained (FP32/16 precision, input size, different conv, activation function...)<br>

## requirements

TensorRT 7.0 (you need to install tensorrt first)<br>
Cuda 10.2<br>
Python3.7<br>
opencv 4.4<br>
cmake 3.18<br>

## Jetson Nano environment preparision


# cuda for tensorrt


```
export CUDA_HOME=/usr/local/cuda # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
```
Then run `source ~/.bashrc`. You will be run `nvcc --version` at any directory.
Reference: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup

### Python 3.7 

```
sudo apt install python3 python3-dev python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3

sudo python3 -m pip install --upgrade pip

sudo python3.7 -m pip install virtualenvwrapper
sudo pip3 install --user virtualenvwrapper

# Add the following lines to `~/.bashrc`:
#   export PATH=/home/nano/.local/bin${PATH:+:${PATH}}

#   # virtualenvwrapper
#   # See https://virtualenvwrapper.readthedocs.io/en/latest/install.html#quick-start
#   export WORKON_HOME=$HOME/.virtualenvs
#   export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
#   source/usr/local/bin/virtualenvwrapper.sh
```

### OpenCV 4.5
```
# Create a directory for third-party source code
mkdir ~/third_party
```
#### Temporarily increase swap space
```
cd ~/third_party
git clone https://github.com/JetsonHacksNano/resizeSwapMemory
cd resizeSwapMemory
zramctl
# Indicates that there is currently 2 GB of swap space
# Increase swap size to 8 GB
./setSwapMemorySize.sh -g 8
sudo systemctl reboot
```
Then follow the link below to install opencv 4.5:
https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html (no need to swap memory again)


#### Reset power mode
```
sudo jetson_clocks --restore ~/Downloads/l4t_dfs.conf
rm ~/Downloads/l4t_dfs.conf
```

#### Revert swap space change
```
cd ~/third_party/resizeSwapMemory
./setSwapMemorySize.sh -g 2
sudo systemctl reboot
zramctl
# Indicates that there is currently 2 GB of swap space
```



### CMake 3.18
```
sudo apt remove cmake # Not necessary
sudo apt install libcurl4-openssl-dev
wget https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz
tar xf cmake-3.18.0.tar.gz
cd cmake-3.18.0
./bootstrap --system-curl
make -j4
sudo make install
#### Verification
cmake --version
#### Deleted CMake source code once finished
```

# train .pth file and convert .wts

## create env

```
pip install -r requirements.txt
```

## train .pth file

train your dataset by following [pytorch-unet](https://github.com/milesial/Pytorch-UNet) and generate .pth file.<br>

## convert .wts

Please change the checkpoint file path in `gen_wts.py`.

```
python3 gen_wts.py
```
run gen_wts from utils folder, and move it to project folder<br>

# generate engine file and infer

You need to change the hardcode directory in unet.cpp for the .wts model

## create build folder in project folder
```
mkdir build
```

## make file, generate exec file
```
cd build
cmake ..
make
```

## generate TensorRT engine file and infer image
Go back to the root directory, 
then run:
```
./build/unet -s
```
then a unet exec file will generated, you can use unet -d to infer files in a folder<br>
```
unet -d ../samples
```

# efficiency
the speed of tensorRT engine is much faster

 pytorch | TensorRT FP32 | TensorRT FP16
 ---- | ----- | ------  
 816x672  | 816x672 | 816x672 
 58ms  | 43ms (batchsize 8) | 14ms (batchsize 8) 


# Further development

1. add INT8 calibrator<br>
2. add custom plugin<br>
etc
