from hyperstream import HyperStream, Workflow
from hyperstream import TimeInterval
from hyperstream.utils import UTC
import hyperstream

import numpy as np

from datetime import datetime, timedelta
from dateutil.parser import parse

from sklearn.linear_model import  SGDClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle

def main(epochs=50):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = load_iris()
    data_tool = hs.plugins.sklearn.tools.dataset(data, shuffle=True,
                                                 epochs=epochs)
    data_stream = M.get_or_create_stream('dataset')

    classifier = SGDClassifier(loss="log", penalty="l2")
    classifier_tool = hs.plugins.sklearn.tools.classifier(classifier)
    classifier_stream = M.get_or_create_stream('classifier')

    now = datetime.utcnow().replace(tzinfo=UTC)
    now = (now - timedelta(hours=1)).replace(tzinfo=UTC)
    before = datetime.utcfromtimestamp(0).replace(tzinfo=UTC)
    ti = TimeInterval(before, now)

    data_tool.execute(sources=[], sink=data_stream, interval=ti)

    #for key, value in data_stream.window():
    #    print('[%s]: %s' % (key, value))

    classifier_tool.execute(sources=[data_stream], sink=classifier_stream,
                            interval=ti)

    scores = []
    for key, value in classifier_stream.window():
        # print('[%s]: %s' % (key, value))
        scores.append(value['score'])

    scores = np.array(scores).reshape(epochs, -1)
    print(scores.mean(axis=1).round(decimals=2))

def test_classifier(epochs=100):
    classifier = SGDClassifier(loss="hinge", penalty="l2")
    data = load_iris()
    d_x = data.data
    classes = [0, 1, 2]
    #d_y = label_binarize(data.target, classes)
    d_y = data.target
    d_x, d_y = shuffle(d_x, d_y)
    classifier = OneVsRestClassifier(classifier)
    first = True
    scores = []
    for i in range(epochs):
        for x, y in zip(d_x, d_y):
            x = x.reshape(1,-1)
            y = y.reshape(1,-1)
            if first:
                classifier.partial_fit(x, y, classes)
                first = False
            else:
                classifier.partial_fit(x, y)
            scores.append(classifier.score(d_x, d_y))

    scores = np.array(scores).reshape(epochs, -1)
    print(scores.mean(axis=1).round(decimals=2))

if __name__=='__main__':
    main()
    #test_classifier()
