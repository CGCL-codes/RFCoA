# RFCOA

The source code of AAAI 2025 paper "Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature".

## Setup Environment

Our project is based on [Horizon2333/imagenet-autoencoder](https://github.com/Horizon2333/imagenet-autoencoder) , the setup environment and configurations can be found in [Autoencoder.md](./Autoencoder.md).

## Data Preparation

Download the dataset ImageNet ILSVRC 2012, the path is as follows:

```
Source Code
├── data
│   ├── ILSVRC2012_img_train.tar
    ├── ILSVRC2012_img_test.tar
    ├── ILSVRC2012_img_val.tar
    ├── ILSVRC2012_devkit_t12.tar.gz
```

## Get the autoencoder

You can train an autoencoder follow the guidance in  [Autoencoder.md](./Autoencoder.md), or download a pertrained autoencoder from the [link](https://drive.google.com/file/d/1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8/view?usp=sharing).

## Extract the Robust Features of Target Class

According to our method, you should extract the robust features of the target class first. 

```
python extract.py
```

## Generate Adversarial Examples

After extract the robust features, you can generate the adversarial examples by running:

```
python attack.py --target xxx
```
