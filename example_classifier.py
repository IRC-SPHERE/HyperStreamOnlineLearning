import numpy as np
import argparse

from datetime import datetime, timedelta

from sklearn import datasets
from sklearn.linear_model import SGDClassifier

from hyperstream import HyperStream, TimeInterval
from hyperstream.utils import UTC


def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', type=str, default='iris',
                        help='''Dataset to use. Working options: iris,
                                breast_cancer, wine, digits''')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to run the classifier')

    return parser.parse_args()


def main(dataset, epochs):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = getattr(datasets, 'load_{}'.format(dataset))()
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

    print("Example of a data stream")
    key, value = data_stream.window().iteritems().next()
    print('[%s]: %s' % (key, value))

    classifier_tool.execute(sources=[data_stream], sink=classifier_stream,
                            interval=ti)

    scores = []
    for key, value in classifier_stream.window():
        scores.append(value['score'])

    scores = np.array(scores).reshape(epochs, -1)
    print(scores.mean(axis=1).round(decimals=2))


if __name__ == '__main__':
    arguments = get_arguments()
    main(**vars(arguments))
