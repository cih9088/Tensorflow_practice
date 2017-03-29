# [InfoGAN](https://arxiv.org/abs/1606.03657)
Information Maximizing Generative Adversarial Networks

## Prerequisite
1. [Tensorflow >= r1.0](https://www.tensorflow.org)
2. [OpenCV](http://opencv.org)

## Usage
To train a model
```
python main.py --data mnist --log_dir results_mnist --is_train
```

To test a existing model
```
python main.py --data mnist --log_dir results_mnist
```
