# Green Lab

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

Install Green Lab:

```
pip install greenlab
```

Install dependencies:

```
pip install -r requirements.txt
```
or if you are using conda:
```
conda env create -f environment.yml
```

## Quick Start

A toy example of using Green Lab to train a model on the CIFAR-10 dataset is provided in [src/cifar.ipynb](src/cifar.ipynb).

<!-- ## <a name="GettingStarted"></a>Getting Started -->


## Citing Green Lab

If you use Green Lab in your research, please use the following BibTeX entry.

```
@inproceedings{chen2021defakehop,
  title={Defakehop: A light-weight high-performance deepfake detector},
  author={Chen, Hong-Shuo and Rouhsedaghat, Mozhdeh and Ghani, Hamza and Hu, Shuowen and You, Suya and Kuo, C-C Jay},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```