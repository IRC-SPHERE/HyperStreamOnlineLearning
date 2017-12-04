# HyperStreamOnlineLearning

Online Learning for [HyperStream](https://github.com/IRC-SPHERE/HyperStream).

## Table of Contents
  - [Requirements](#requirements)
  - [Unittest](#unittest)
  - [Run a simple classification test](#run-a-simple-classification-test)
  - [Example with Keras](#example-with-keras)
  - [Example of Anomaly detection](#example-of-anomaly-detection)
  - [Example of Incremental PCA](#example-of-incremental-pca)
  - [Example of an Autoencoder with Keras](#example-of-an-autoencoder-with-keras)

## Requirements

Running any example of HyperStream requires a MongoDB server. The configuration
of the host and ports of the MongoDB server can be changed in the file
`hyperstream_config.json` if needed.

To download the code and install the requirements ussing virtualenvironment just do the following:

```bash
git clone git@github.com:IRC-SPHERE/HyperStreamOnlineLearning.git
cd HyperStreamOnlineLearning
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

If you want to run the examples with Keras you will need to install some additional dependences. Once the previous requirements have been installed you can install the rest with the following command:

```bash
pip install -r keras_requirements.txt
```

Once the MongoDB server is up and running and you have installed all the Python requirements you can try to run the some fo the following examples:

## Unittest

To test if the code is working you can run a nosetest

```bash
nosetest
```

You should get something of the style

```bash
.
----------------------------------------------------------------------
Ran 1 test in 0.585s

OK
```

## Run a simple classification test

The following code should finish with the scores of a model trained on Iris dataset.

```bash
python example_classifier.py --dataset iris --classifier SGDClassifier --epochs 20 --seed 42
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
Test scores per epoch during training
[ 0.59  0.65  0.68  0.71  0.72  0.72  0.65  0.73  0.69  0.72  0.71  0.73
  0.72  0.72  0.72  0.72  0.72  0.76  0.75  0.77]
```

## Example with Keras

There is an additional example using Keras to specify Multilayer Perceptrons or
Logistic regression. To run the example use these steps

```bash
python example_classifier_keras.py --dataset digits --classifier mlp30ds40m --epochs 20 --seed 42
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
[ 0.48  0.75  0.84  0.89  0.89  0.9   0.91  0.91  0.92  0.91  0.92  0.94
  0.93  0.94  0.93  0.93  0.94  0.94  0.95  0.94]
```

## Example of Anomaly detection

Example of a model that trains only with the input space and predicts if the
test data has been drawn from the same distribution. In this case using a
Multivariate-Gaussian to estimate the density of the training data.

```bash
python example_anomalies.py --dataset iris --model Gaussian --epochs 1 --seed 42 -b 2
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
[  nan   nan   nan  0.92  0.93  0.57  0.52  0.91  0.81  0.41  0.84  0.53
  0.77  0.88  0.95  0.79  0.58  0.84  0.69  0.82  0.63  0.55  0.99  0.9
  0.89  0.65  0.61  0.83  0.87  0.86  0.78  0.75  0.74  0.66  0.84  0.92
  0.81  0.81]
```

## Example of Incremental PCA

```bash
python example_incremental_pca.py --dataset digits --components 10 --epochs 10 --seed 42 -b 100
```

At the end of the training you should see the mean squared error of the
reconstruction using the principal components of the PCA.

```Python
Test scores per minibatch (cyclic)
[[ 5.8   5.11  5.54  5.33  5.59  5.08  5.35  5.14  5.04  5.18  4.79  5.42
   5.24  5.48  5.06  5.33  5.13  5.01  5.2   4.76  5.43  5.2   5.45  5.09
   5.36  5.06  4.99  5.26  4.73  5.41  5.22  5.4   5.12  5.35  5.08  4.93
   5.29  4.73  5.44  5.15  5.37  5.15  5.35  5.08  4.94  5.27  4.77  5.43
   5.13  5.37  5.26  5.23  5.07  4.94  5.29  4.78  5.41  5.13  5.35  5.27
   5.24  5.06  4.93  5.28  4.83  5.39  5.18  5.28  5.27  5.28  5.03  4.94
   5.28  4.84  5.43  5.14  5.25  5.3   5.26  5.06  4.89  5.3   4.85  5.4
   5.17  5.21  5.32  5.23  5.08  5.04]]
```

![pca_inverse_transform.svg][pca_inverse_transform.svg]

## Example of an Autoencoder with Keras

```bash
python example_autoencoder_keras.py --dataset digits --architecture auto30ns10ns2ns_10ns30ns --epochs 100 --seed 42 -b 10 --learning-rate 0.1
```

Will output the following

```Python
_________________________________________________________________
Layer (type)                 Output Shape              Param #                         [20/9638]
=================================================================
dense_1 (Dense)              (None, 30)                1950
_________________________________________________________________
batch_normalization_1 (Batch (None, 30)                120
_________________________________________________________________
activation_1 (Activation)    (None, 30)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                310
_________________________________________________________________
batch_normalization_2 (Batch (None, 10)                40
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
_________________________________________________________________
batch_normalization_3 (Batch (None, 2)                 8
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0
=================================================================
Total params: 2,450
Trainable params: 2,366
Non-trainable params: 84
_________________________________________________________________
None
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_4 (Dense)              (None, 10)                30
_________________________________________________________________
batch_normalization_4 (Batch (None, 10)                40
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
_________________________________________________________________
dense_5 (Dense)              (None, 30)                330
_________________________________________________________________
batch_normalization_5 (Batch (None, 30)                120
_________________________________________________________________
activation_5 (Activation)    (None, 30)                0
_________________________________________________________________
dense_6 (Dense)              (None, 64)                1984
=================================================================
Total params: 2,504
Trainable params: 2,424
Non-trainable params: 80
_________________________________________________________________
None
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential_1 (Sequential)    (None, 2)                 2450      
_________________________________________________________________
sequential_2 (Sequential)    (None, 64)                2504      
=================================================================
Total params: 4,954
Trainable params: 4,790
Non-trainable params: 164
_________________________________________________________________
None
2017-12-04 15:49:37.732891: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Test scores per minibatch (cyclic)
[[ 65.96  58.56  60.33 ...,  13.54  10.89  12.94]]
```

![autoencoder_auto30ns10ns2ns_10ns30ns.svg][autoencoder_auto30ns10ns2ns_10ns30ns.svg]
