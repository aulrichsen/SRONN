# Self-ONN #

Alexander Ulrichsen's implementation of Self-ONN from paper: https://arxiv.org/pdf/2004.11778.pdf

## File descriptors: ##

Self_ONN:           contains Operator_Layer class which is essentially the ONN version of nn.Conv2d()
models:             contains basic ONN which has an embedding output
test_ONN_learning:  creates random input and random (non-linear) target funtion and trains network to see if weights from target function can be learned.

## Quick start ##

* 1) install pytorch: https://pytorch.org
* 2) Create a model architecture with the Operator_Layer, or use example model in models.py file
    * It is reccomended that you follow each Operator_Layer with a Tanh layer to bound data between -1 and 1
* 3) Train using standard pytorch training algorithm
    * Tutorial on how to do this with a CNN can be found here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## General Concept behind Operator_Layer ##

Uses MacLaurin series approximations to create learnable non-linear convolutional filters.

![alt text](https://github.com/aulrichsen/Self-ONN/blob/main/MacLaurin_Series.png?raw=true)

Achieves this by taking all input channels and raising them to all the powers up to the specified q_order (the approximation order) and applying regular convolution to all the original and raised channels. Note, there is no bias in the convolution and a seperate bias layer is applied after the convolution as this is how the MacLaurin approximation is performed.

## Potential Ideas for Performance improvements ##

* 1) Use lower learning rates for filters on the higher order componenets.
    * E.g y^1 weights lr=0.01, y^2 weights lr=0.001, y^3 weights lr=0.0001 ...
    * This may help with training as higher order components may cause instability.

* 2) Replace Tanh bounding layer with something like L2 norm or Batch Norm
