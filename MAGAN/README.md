# [MAGAN](https://arxiv.org/abs/1704.03817)
Margin Adaptation for Generative Adversarial Networks

## Differences
1. Batch Normalization is used for all the networks
    Leaky RELU in the discriminator shows  instable behaviour

## Prerequisite
1. [Tensorflow >= r1.0](https://www.tensorflow.org)
2. [OpenCV](http://opencv.org)

## Usage
To train a model
```
python main.py --data mnist --log_dir results_mnist --is_train --z_dim 50
python main.py --data cifar10 --log_dir results_cifar10 --is_train --z_dim 320
```

To test a existing model
```
python main.py --data mnist --log_dir results_mnist
python main.py --data cifar10 --log_dir results_cifar10
```
