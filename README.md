# Membership-Inference-Attacks-Against-Object-Detection-Models


## Requirements

* python
* Chainer 
* Pytorch

## Training Shadow and Target Models
To train object detection models, please run the following files.

```
# For SSD300 using VOC dataset
python train_Chainer_ssd_voc_shadow.py  [gpu_id]
python train_Chainer_ssd_voc_target.py  [gpu_id]

```
python mia_train_attacker_.py  config.py

To run the code, please execute 
```
python mia_train_attacker_.py  config.py
```
