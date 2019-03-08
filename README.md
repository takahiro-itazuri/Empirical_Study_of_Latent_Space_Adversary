# Adversarial Examples Projects
This is a base repository of adversarial examples projects.

## Setup
### 1. Install Packages
```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets
Download the [ImageNet dataset](http://image-net.org/download) (ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar) into ImageNet directory (data/imagenet) and move validation images to labeled subfolders.
```bash
cd data
mkdir imagenet
bash extract.sh # extract tar files
bash valprep.sh # move validation images to 
```
