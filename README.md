# Install

```
pip install openmim
pip install numpy<2
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

mim install mmengine --constraint constraints.txt
mim install mmdet

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

pip install -r requirements.txt
```

## RTX PRO 6000

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install ninja

pip install openmim
pip install numpy<2
pip install torch==2.8.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu128

mim install mmengine --constraint constraints.txt
mim install mmdet

pip install mmcv==2.1.0

pip install -r requirements.txt
```

# Train

```
mim train mmdet .\config\mask_rcnn_manga.py
```

## RTX PRO 6000

```
unset TMUX
tmux new -s training -d
tmux attach -t training

mim train mmdet ./config/mask_rcnn_manga_6000.py
```
