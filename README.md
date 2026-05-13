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

# Train

```
mim train mmdet .\config\mask_rcnn_manga.py
```
