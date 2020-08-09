# XNOR Net (Original Paper) - PyTorch 

This repository implements the popular paper: **XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks**: https://arxiv.org/abs/1603.05279.



The project is organized as follows.

  - *models* folder contains CNN models (simple mlp, Network-in-Network, LeNet5)
  - *xnor_classifier.py* contains the test and train procedures
  - *models/binary_layers.py* contains the XNOR layers implementation (binary activation, binary conv and fully-connected layers, gradient update)
  - *yml* folder contains configuration files with hyperparameters
  - *main.py* represents the entry file

### Installation

All packages are in *requirement.txt*
Install the dependencies:

```sh
pip install -r requirements.txt
```
### Basic usage
```sh
$ python main.py app:{yml_file}
```
### Example 
Network-in-Network on CIFAR10 dataset. All hyper parameters are in .yml file. 
```sh
$ python main.py app:yml/nin_cifar10.yml
```
## Related Applications
  - CoopNet: Cooperative Convolutional Neural Network for Low-Power MCUs https://arxiv.org/abs/1911.08606
  - TentacleNet: A Pseudo-Ensemble Template for Accurate Binary Convolutional Neural Networks https://arxiv.org/abs/1912.10103

License
----

MIT




