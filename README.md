# Criss-Cross Attention for Semantic Segmentation (CCNet)

This is a Jittor implementation of CCNet proposed by paper "CCNet: Criss-Cross Attention for Semantic Segmentation". We conduct experiment on Cityscapes and ADE20K segmentation dataset, and also provided access to VOC dataset. 

Cityscapes, ADE20K and VOC are widely-used datasets for fine-grained visual segmentation task.

We also re-implement the original CCNet with a VAN backbone. Evaluation results on Cityscapes and ADE20K dataset could be found below.



## Overview of the Paper

In this section, we refer to the original [github repository](https://github.com/speedinghzl/CCNet/) to give a brief introduction to their work.

### Introduction

[![motivation of CCNet](https://user-images.githubusercontent.com/4509744/50546460-7df6ed00-0bed-11e9-9340-d026373b2cbe.png)](https://user-images.githubusercontent.com/4509744/50546460-7df6ed00-0bed-11e9-9340-d026373b2cbe.png) Long-range dependencies can capture useful contextual information to benefit visual understanding problems. In this work, the author propose a Criss-Cross Network (CCNet) for obtaining such important information through a more effective and efficient way. Concretely, for each pixel, our CCNet can harvest the contextual information of its surrounding pixels on the criss-cross path through a novel criss-cross attention module. By taking a further recurrent operation, each pixel can finally capture the long-range dependencies from all pixels. Overall, our CCNet is with the following merits:

- **GPU memory friendly**
- **High computational efficiency**
- **The state-of-the-art performance**



### Architecture

[![Overview of CCNet](https://user-images.githubusercontent.com/4509744/50546462-851dfb00-0bed-11e9-962a-bffab2401997.png)](https://user-images.githubusercontent.com/4509744/50546462-851dfb00-0bed-11e9-962a-bffab2401997.png) Overview of the proposed CCNet for semantic segmentation. The proposed recurrent criss-cross attention takes as input feature maps **H** and output feature maps **H''** which obtain rich and dense contextual information from all pixels. Recurrent criss-cross attention module can be unrolled into R=2 loops, in which all Criss-Cross Attention modules share parameters.



### Visualization of the attention map

[![Overview of Attention map](https://user-images.githubusercontent.com/4509744/50546463-88b18200-0bed-11e9-9f87-c74327c11a68.png)](https://user-images.githubusercontent.com/4509744/50546463-88b18200-0bed-11e9-9f87-c74327c11a68.png) To get a deeper understanding of our RCCA, the author visualize the learned attention masks as shown in the figure. For each input image, they select one point (green cross) and show its corresponding attention maps when **R=1** and **R=2** in columns 2 and 3 respectively. In the figure, only contextual information from the criss-cross path of the target point is capture when **R=1**. By adopting one more criss-cross module, ie, **R=2** the RCCA can finally aggregate denser and richer contextual information compared with that of **R=1**. Besides, they observe that the attention module could capture semantic similarity and long-range dependencies.




## Reproduction Results

The following results are based on our model, which is implemented with [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/). The details of the model is almost the same with the original Pytorch version.

| Model              | Dataset    | meanIoU (%) |
| ------------------ | ---------- | ----------- |
| CCNet (resnet-101) | Cityscapes | 79.2        |
| CCNet (resnet-101) | ADE20K     | 43.51       |
| CCNet (VAN-large)  | Cityscapes | 80.4        |
| CCNet (VAN-large)  | ADE20K     | 44.83       |

All pretrain models could be downloaded from [here](https://cloud.tsinghua.edu.cn/d/8490a3718b9245c5b19b/).



## Instructions for Code

### Requirement

To install Jittor==1.3.6.10, please refer to https://cg.cs.tsinghua.edu.cn/jittor/download/.

4 x 32G GPUs (e.g. V100)

Python 3.9

CUDA 11.6

### Datasets and pretrained model

Please download cityscapes dataset, ADE20K dataset, VOC dataset and unzip them into `YOUR_DATASET_PATH`.

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

Please download [VAN pretrained model](https://github.com/Visual-Attention-Network/VAN-Classification), and also put it into `dataset` folder.

### Training and Evaluation

#### Training script

CCNet on ADE20K.
```bash
python train.py --data-dir ${YOUR_DATASET_PATH} --restore-from ./dataset/resnet101-imagenet.pth --start 0 --max-iter 60000 --random-mirror --random-scale --learning-rate 1e-2 --input-size 512 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --save-pred-every 2000 --recurrence 2 --model ccnet_large --dataset ade20k --num-classes 151
```

CCNet on Cityscapes.
```bash
python train.py --data-dir ${YOUR_DATASET_PATH} --restore-from ./dataset/resnet101-imagenet.pth --start 0 --max-iter 60000 --random-mirror --random-scale --learning-rate 1e-2 --input-size 512 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --save-pred-every 2000 --recurrence 2 --model ccnet_large --dataset cityscape --num-classes 19
```

CCNet (with VAN backbone) on ADE20K.
```bash
python train.py --data-dir ${YOUR_DATASET_PATH} --restore-from /dataset/van_large.pth --start 0 --max-iter 60000 --random-mirror --random-scale --learning-rate 1e-2 --input-size 512 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --save-pred-every 2000 --recurrence 2 --model van --dataset ade20k --num-classes 151 --van-size van_large
```

CCNet (with VAN backbone) on Cityscapes.
```bash
python train.py --data-dir ${YOUR_DATASET_PATH} --restore-from /dataset/van_large.pth --start 0 --max-iter 60000 --random-mirror --random-scale --learning-rate 1e-2 --input-size 512 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --save-pred-every 2000 --recurrence 2 --model van --dataset cityscape --num-classes 19 --van-size van_large
```

#### Evaluation script

Cityscapes.

```bash
python evaluate.py --restore-from YOUR_CHECKPOINT --input-size 512 --recurrence 2 --model [ccnet_large,van] --dataset cityscape --num-classes 19 --ignore-label 255
```

ADE20K
```bash
python evaluate.py --restore-from YOUR_CHECKPOINT --input-size 512 --recurrence 2 --model [ccnet_large,van] --dataset cityscape --num-classes 151 --ignore-label 0
```



## Citation

```
@article{huang2020ccnet,
  author={Huang, Zilong and Wang, Xinggang and Wei, Yunchao and Huang, Lichao and Shi, Humphrey and Liu, Wenyu and Huang, Thomas S.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={CCNet: Criss-Cross Attention for Semantic Segmentation}, 
  year={2020},
  month={},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantic Segmentation;Graph Attention;Criss-Cross Network;Context Modeling},
  doi={10.1109/TPAMI.2020.3007032},
  ISSN={1939-3539}}

@article{huang2018ccnet,
    title={CCNet: Criss-Cross Attention for Semantic Segmentation},
    author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle={ICCV},
    year={2019}}

@article{guo2022visual,
  title={Visual attention network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}}

@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={12},
  pages={1--21},
  year={2020},
  publisher={Springer}}
```



## Acknowledgment

Our implementation is mainly based on [CCNet](https://github.com/speedinghzl/CCNet/) and [VAN](https://github.com/Visual-Attention-Network). We use [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) as our deep learning framework. Thanks for their authors.