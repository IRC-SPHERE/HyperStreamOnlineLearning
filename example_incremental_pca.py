import numpy as np
import argparse

from datetime import datetime, timedelta

from sklearn import datasets, linear_model

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error as mse

from hyperstream import HyperStream, TimeInterval
from hyperstream.utils import UTC


class MyIncrementalPCA(IncrementalPCA):
    def score(self, x, pred):
        return mse(x, pred)


def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', type=str, default='iris',
                        help='''Dataset to use. Working options: iris,
                                breast_cancer, wine, digits''')
    parser.add_argument('-c', '--components', type=int, default=2,
                        help='Number of principal components to learn')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to run Incremental PCA')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Seed for the data shuffle')
    parser.add_argument('-b', '--batchsize', type=int, default=1,
                        help='Batch size during training')

    return parser.parse_args()


def main(dataset, components, epochs, seed, batchsize):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = getattr(datasets, 'load_{}'.format(dataset))()
    data_tool = hs.plugins.sklearn.tools.dataset(data, shuffle=True,
                                                 epochs=epochs, seed=seed)
    data_stream = M.get_or_create_stream('dataset')

    print('components = {}'.format(components))
    model = MyIncrementalPCA(n_components=components)
    print(model.n_components)
    unsupervised_tool = hs.plugins.sklearn.tools.unsupervised(model)
    unsupervised_stream = M.get_or_create_stream('unsupervised')

    now = datetime.utcnow().replace(tzinfo=UTC)
    now = (now - timedelta(hours=1)).replace(tzinfo=UTC)
    before = datetime.utcfromtimestamp(0).replace(tzinfo=UTC)
    ti = TimeInterval(before, now)

    data_tool.execute(sources=[], sink=data_stream, interval=ti)

    print("Example of a data stream")
    key, value = data_stream.window().iteritems().next()
    print('[%s]: %s' % (key, value))

    mini_batch_tool = hs.plugins.sklearn.tools.minibatch(batchsize=batchsize)
    mini_batch_stream = M.get_or_create_stream('mini_batch')
    mini_batch_tool.execute(sources=[data_stream], sink=mini_batch_stream,
                            interval=ti)

    unsupervised_tool.execute(sources=[mini_batch_stream],
                              sink=unsupervised_stream,
                              interval=ti)

    scores = []
    for key, value in unsupervised_stream.window():
        scores.append(value['score'])

    # The data is repeated the number of epochs. This makes the mini-batches to
    # cycle and contain data from the begining and end of the dataset. This
    # makes possible that the number of scores is not divisible by epochs.
    if batchsize == 1:
        print("Test scores per epoch")
        scores = np.array(scores).reshape(epochs, -1)
        print(scores.mean(axis=1).round(decimals=2))
    else:
        scores = np.array(scores).reshape(1,-1)
        print("Test scores per minibatch (cyclic)")
        print(scores.round(decimals=2))


if __name__ == '__main__':
    arguments = get_arguments()
    main(**vars(arguments))
