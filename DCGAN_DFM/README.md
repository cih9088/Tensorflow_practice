# [DCGAN_DFM](https://openreview.net/pdf?id=S1X7nhsxl)
Improving Generative Adversarial Networks with Denoising Feature Matching (TensorFlow)

## Prerequisite
1. [Tensorflow >= r1.0](https://www.tensorflow.org)
2. [OpenCV](http://opencv.org)

## Usage
To train a model
```
python main.py --data mnist --log_dir results_mnist --is_train
python main.py --data cifar10 --log_dir results_cifar10 --is_train
```

To test a existing model
```
python main.py --data mnist --log_dir results_mnist
python main.py --data cifar10 --log_dir results_cifar10
```
