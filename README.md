# HyperStreamOnlineLearning

Online Learning for [HyperStream](https://github.com/IRC-SPHERE/HyperStream).

## Run a test

Running any example of HyperStream requires a MongoDB server. The configuration
of the host and ports of the MongoDB server can be changed in the file
`hyperstream_config.json` if needed.

Once the MongoDB server is up and running you can try to run the following code
that should finish with the scores of a model trained on Iris dataset.

```bash
git clone git@github.com:IRC-SPHERE/SklearnHyperStream.git
cd SklearnHyperStream
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
python example_classifier.py --dataset iris --classifier SGDClassifier --epochs 20 --seed 42
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
Test scores per epoch during training
[ 0.52  0.63  0.68  0.73  0.72  0.73  0.69  0.72  0.73  0.69  0.75  0.71
  0.75  0.76  0.71  0.69  0.76  0.69  0.71  0.69]
```

## Example with Keras

There is an additional example using Keras to specify Multilayer Perceptrons or
Logistic regression. To run the example use these steps

```bash
git clone git@github.com:IRC-SPHERE/SklearnHyperStream.git
cd SklearnHyperStream
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -r keras_requirements.txt
python example_classifier_keras.py --dataset digits --classifier mlp30ds40m --epochs 20 --seed 42
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
[ 0.67  0.88  0.9   0.92  0.93  0.92  0.92  0.93  0.93  0.93  0.94  0.95
  0.95  0.94  0.94  0.94  0.94  0.94  0.94  0.94]
```

## Example of Anomaly detection

Example of a model that trains only with the input space and predicts if the
test data has been drawn from the same distribution. In this case using a
Multivariate-Gaussian to estimate the density of the training data.

```bash
git clone git@github.com:IRC-SPHERE/SklearnHyperStream.git
cd SklearnHyperStream
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -r keras_requirements.txt
python example_anomalies.py --dataset iris --model Gaussian --epochs 1 --seed 42 -b 2
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
[  nan   nan  1.    1.    0.67  1.    0.8   0.88  0.97  0.98  0.85  1.
  0.85  0.95  0.83  0.94  0.65  0.8   0.95  0.79  0.91  0.54  0.88  0.63
  0.63  0.94  0.87  0.82  0.72  0.78  0.98  0.82  0.76  0.79  0.65  0.87
  0.89  0.85]
```
