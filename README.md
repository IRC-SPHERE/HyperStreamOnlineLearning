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
python example_classifier.py
```
