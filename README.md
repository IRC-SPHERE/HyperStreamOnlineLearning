# HyperStreamOnlineLearning

Online Learning for [HyperStream](https://github.com/IRC-SPHERE/HyperStream).

## Table of Contents
  - [Requirements](#requirements)
  - [Run a simple test](#run-a-simple-test)
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

## Run a simple test

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

## Example of an Autoencoder with Keras

```bash
python example_autoencoder_keras.py --dataset digits --architecture auto40s20s10s --epochs 10 --seed 42 -b 100
```

Will output the following

```Python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 40)                2600
_________________________________________________________________
activation_1 (Activation)    (None, 40)                0
_________________________________________________________________
dense_2 (Dense)              (None, 20)                820
_________________________________________________________________
activation_2 (Activation)    (None, 20)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0
=================================================================
Total params: 3,630
Trainable params: 3,630
Non-trainable params: 0
_________________________________________________________________
None
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_4 (Dense)              (None, 10)                110
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
_________________________________________________________________
dense_5 (Dense)              (None, 20)                220
_________________________________________________________________
activation_5 (Activation)    (None, 20)                0
_________________________________________________________________
dense_6 (Dense)              (None, 40)                840
_________________________________________________________________
dense_7 (Dense)              (None, 64)                2624
=================================================================
Total params: 3,794
Trainable params: 3,794
Non-trainable params: 0
_________________________________________________________________
None
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential_1 (Sequential)    (None, 10)                3630
_________________________________________________________________
sequential_2 (Sequential)    (None, 64)                3794
=================================================================
Total params: 7,424
Trainable params: 7,424
Non-trainable params: 0
_________________________________________________________________
None
Test scores per minibatch (cyclic)
[[ 58.15  57.16  57.83  59.39  56.94  56.84  56.45  55.91  55.95  54.46
   53.64  54.08  55.71  53.21  53.16  52.72  52.25  52.09  50.4   49.72
   50.25  51.62  49.02  48.96  48.54  47.99  47.7   46.05  45.28  45.9
   46.92  44.46  44.27  43.95  43.35  42.86  41.41  40.64  41.19  41.7
   39.84  39.5   39.02  38.45  38.18  36.63  35.98  36.3   36.71  35.13
   34.8   34.37  33.7   33.57  32.07  31.52  31.79  31.83  30.78  30.42
   29.96  29.55  29.17  28.03  27.46  27.77  27.57  26.84  26.63  26.12
   25.8   25.66  24.67  24.17  24.46  24.02  23.63  23.68  23.1   22.87
   22.96  22.2   21.7   22.    21.41  21.39  21.56  20.95  20.82  20.97]]
```
