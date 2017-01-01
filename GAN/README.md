# [GAN](https://arxiv.org/pdf/1406.2661v1.pdf)
Generative Adversarial Network (TensorFlow)

## Prerequiste
1. [PrettyTensor](https://github.com/google/prettytensor)
2. [Fuel](https://github.com/mila-udem/fuel)
3. [OpenCV](http://opencv.org)

## How to train
```
python multiGPU-GAN.py --batch_size 100 --n_gpu 1 --data mnist
python multiGPU-GAN.py --batch_size 50 --n_gpu 2 --data cifar10
```
