import numpy as np
import argparse

from datetime import datetime, timedelta

from sklearn import datasets, linear_model

from hyperstream import HyperStream, TimeInterval
from hyperstream.utils import UTC

class GaussianEstimator(object):
    def __init__(self):
        self.mu = None
        self.cov = None
        self.N = 0

    def fit(self, x):
        N = x.shape[1]
        mu = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)

        if self.N is 0:
            self.N = N
            self.mu = mu
            self.k = len(mu)
            self.cov = cov
        else:
            self.mu = np.true_divide((self.mu * self.N) + (mu * N), self.N + N)
            self.cov = np.true_divide((self.cov * self.N) + (cov * N), self.N + N)
            self.N += N

    def likelihood(self, x):
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x):
        x_mu = x - self.mu
        # a = np.array([[1, 2]])
        # b = np.array([[1, 2],[3,4]])
        # np.inner(np.inner(a, b.T), a)
        inverse = np.linalg.pinv(self.cov)
        exp = np.array([np.inner(np.inner(a, inverse.T), a) for a in x_mu])
        return - 0.5 * (
                    np.log(np.linalg.det(self.cov))
                    + exp \
                    + self.k * np.log(2*np.pi)
                    )

def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model', type=str,
                        default='Gaussian',
                        help='''Model to use. Working options:
                                Gaussian''')
    parser.add_argument('-d', '--dataset', type=str, default='iris',
                        help='''Dataset to use. Working options: iris,
                                breast_cancer, wine, digits''')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to run the model')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Seed for the data shuffle')
    parser.add_argument('-b', '--batchsize', type=int, default=2,
                        help='Batch size during training')

    return parser.parse_args()


def main(dataset, model, epochs, seed, batchsize):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = getattr(datasets, 'load_{}'.format(dataset))()
    data_tool = hs.plugins.sklearn.tools.dataset(data, shuffle=True,
                                                 epochs=epochs, seed=seed)
    data_stream = M.get_or_create_stream('dataset')

    if model == 'Gaussian':
        model = GaussianEstimator()
    else:
        raise(ValueError('Unknown model {}'.format(model)))

    anomaly_detector_tool = hs.plugins.sklearn.tools.anomaly_detector(model)
    anomaly_detector_stream = M.get_or_create_stream('anomaly_detector')

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

    anomaly_detector_tool.execute(sources=[mini_batch_stream],
                                  sink=anomaly_detector_stream, interval=ti)

    probas = []
    for key, value in anomaly_detector_stream.window():
        probas.append(value['proba'])

    # The data is repeated the number of epochs. This makes the mini-batches to
    # cycle and contain data from the begining and end of the dataset. This
    # makes possible that the number of scores is not divisible by epochs.
    probas = np.array(probas)
    print(probas.shape)
    means = np.array([np.nanmean(aux) for aux in probas])
    print(means.shape)
    print("Test probabilities per minibatch (cyclic)")
    print(means.round(decimals=2))


if __name__ == '__main__':
    arguments = get_arguments()
    main(**vars(arguments))
