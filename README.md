# Self-ONN #

Alexander Ulrichsen's implementation of Self-ONN from paper: https://arxiv.org/pdf/2004.11778.pdf

## File descriptors: ##

Self_ONN:           contains Operator_Layer class which is essentially the ONN version of nn.Conv2d()
models:             contains basic ONN which has an embedding output
test_ONN_learning:  creates random input and random (non-linear) target funtion and trains network to see if weights from target function can be learned.

## Quick start ##

1) Intall requirements
```
pip3 install -r requirements.txt
```
2) Download Datasets from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

3) Train a model
```
python3 Training.py --model SRONN --is_residual --dataset PaviaU --scale 2
```

## Training Args ###
 
--model:    Model to use for training.                          | Options: SRCNN, SRONN, SRONN_AEP, SRONN_L2, SRONN_BN\
--dataset:  Dataset to train on.                                | Options: Pavia, Botswana\
--scale:    Super resolution scale factor.\
--epochs:   Number of epochs to train for.\
--lr:       Starting training learning rate.\
--lr_ms:    Epochs at which to decrease learning rate by 10x.\
--opt:      Training optimizer to use.                          | Options: Adam\
--loss:     Training loss function to use.                      | Options: MSE\
--ms:       Number of epochs between logging and display.\
   
All arguments defined within training_setup.py

## General Concept behind Operator_Layer ##

Uses MacLaurin series approximations to create learnable non-linear convolutional filters.

![alt text](https://github.com/aulrichsen/Self-ONN/blob/main/MacLaurin_Series.png?raw=true)

Achieves this by taking all input channels and raising them to all the powers up to the specified q_order (the approximation order) and applying regular convolution to all the original and raised channels. Note, there is no bias in the convolution and a seperate bias layer is applied after the convolution as this is how the MacLaurin approximation is performed.

