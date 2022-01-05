# ProgressFace Face Detector

## Introduction

This code belongs to our ECCV 2020 paper: [ProgressFace: Scale-Aware Progressive Learning for Face Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510341.pdf)

## Data


1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset. 

2. Download annotations (face bounding boxes & five facial landmarks) provided by [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organize the dataset directory under ``data`` as follows: 

```Shell
  data/retinaface/
    train/
      images/
      label.txt
    val/
      images/
      label.txt
    test/
      images/
      label.txt
```

4. Download the evaluation tools of [mkROC](https://github.com/ramanathan831/fddb-evaluation) and [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation) under ``evaluation`` as follows:
```Shell
    evaluation/
        mkROC/
        WiderFace-Evaluation/
```

## Install

1. Install python 2.7
2. pip install -r requirements.txt
3. Install Deformable Convolution V2 operator from [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets) if you use the DCN based backbone.
4. Type ``make`` to build cxx tools.
5. Prepare evaluation environments
```Shell
    cd evaluation/WiderFace-Evaluation
    # make Evaluation tool compiled
    python setup.py build_ext --inplace
```

## Evaluating

```Shell
# Evaluate the pretrained model
python test_widerface.py --network net3 --gpu 0 --prefix ./model/progressface_light  --epoch 0 --mode 1 --output ./wout
# Use evaluation tools to obtain AP on WiderFace
cd evaluation/WiderFace-Evaluation
python evaluation.py -p ../../wout -g ./
```
