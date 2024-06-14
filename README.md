Deep convolution generative adversarial network (DCGAN)
============================================

This repository contains modification of the PyTorch DCGAN tutorial implementation:
```
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
```
which is based on the paper

```
Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional 
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015)
```

Instead of the 64x64 RGB images as in the original tutorial here the 28x28 grayscale images are used 
for training. The data set is downloaded from Kaggle tutorial
```
https://www.kaggle.com/code/songseungwon/pytorch-gan-basic-tutorial-for-beginner
```


### Requirements

The model is implemented in Python 3.11.0 and uses several additional libraries which can be found in
`environments.yaml`.

### Setup

To install this package, simply clone the git repo:

```
git clone ...
cd ...
conda env create -f environment.yaml
conda activate gan-tutorial
```


### Contents

The repository is structured as follows.

* `./results`: resulting images
* `./data`: data

### Dataset

The `./data` subfolder contains csv files with pixel intensity values for about 42000 grayscale 28x28 images. 



Next, you must prepare the dataset for training:
you will need to create pairs of high and low resolution sound patches (typically, about 0.5s in length).
We have included a script called `prep_vctk.py` that does that. 

The output of the data preparation step are two `.h5` archives containing, respectively, the training and validation pairs of high/low resolution sound patches.
You can also generate these by running `make` in the corresponding directory, e.g.
```
cd ./speaker1;
make;
```

### Examples