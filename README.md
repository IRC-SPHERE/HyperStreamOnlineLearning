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
python example_classifier.py --dataset iris --epochs 20 --seed 42
```

At the end of the training you should see the test scores during the training
for each epoch

```Python
Test scores per epoch during training
[ 0.52  0.63  0.68  0.73  0.72  0.73  0.69  0.72  0.73  0.69  0.75  0.71
  0.75  0.76  0.71  0.69  0.76  0.69  0.71  0.69]
```
