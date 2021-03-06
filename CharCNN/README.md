# Character Level CNNs in Keras

This repository contains Keras implementations for Character-level Convolutional Neural Networks for Twitter Events Credbility Classification

The following models have been implemented:
 1. Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). NIPS 2015
 2. Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615). AAAI 2016
 3. Shaojie Bai, J. Zico Kolter, Vladlen Koltun. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf). *ArXiv preprint (2018)*

## Usage

1. Install dependencies (Tensorflow 1.3 and Keras 2.1.3):

```
$ pip install -r requirements.txt
```

2. Specify the training and testing data sources and model hyperparameters in the `config.json` file.

3. Run the main.py file as below:

```sh
$ python main.py --model [model_name]
```

Replace `[model_name]` with either `zhang` or `kim` to run the desired model.
