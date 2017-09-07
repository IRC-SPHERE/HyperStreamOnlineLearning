# SklearnHyperStream

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
python example_classifier.py --dataset digits --classifier mlp30ds40m --epochs 20 --seed 42
```

```Python
[ 0.67  0.88  0.9   0.92  0.93  0.92  0.92  0.93  0.93  0.93  0.94  0.95
  0.95  0.94  0.94  0.94  0.94  0.94  0.94  0.94]
At the end of the training you should see the test scores during the training
for each epoch
```
