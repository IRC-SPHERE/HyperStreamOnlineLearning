import numpy as np
import argparse
import re

from datetime import datetime, timedelta

from sklearn import datasets, linear_model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error as mse

from hyperstream import HyperStream, TimeInterval
from hyperstream.utils import UTC

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

class MyKerasUnsupervised(object):
    """This is a modification of the KerasUnsupervised in order to keep the
    labels with the original values.

    Implementation of the scikit-learn classifier API for Keras.
    """
    def __init__(self, architecture='auto2s', lr=0.1):
        self.model = None
        self.architecture = architecture
        self.lr = lr

    def fit(self, x, classes=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where n_samples in the number of samples
                and n_features is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for X.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """
        if self.model is None:
            self.model = self.create_model(input_dim=x.shape[1],
                                           architecture=self.architecture,
                                           lr=self.lr)
        return self.model.fit(x, x, batch_size=x.shape[0], epochs=1, verbose=0)

    def create_model(self, input_dim=1, optimizer='rmsprop',
                     init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                     nesterov=False , architecture='lr'):
        """
        Parameters
        ----------
        architecture: string: lr, mlp100, mlp100d, mlp100d100d
        """


        previous_layer = input_dim
        self.architecture = re.split('(\d+)', architecture)
        if self.architecture[0] == 'auto':
            encoder = Sequential()
            for i in range(1, len(self.architecture)):
                if self.architecture[i] == 'd':
                    encoder.add(Dropout(0.5))
                elif self.architecture[i] == 's':
                    encoder.add(Activation('sigmoid'))
                elif self.architecture[i] == 'm':
                    encoder.add(Dense(input_size, kernel_initializer=init))
                    encoder.add(Activation('softmax'))
                elif self.architecture[i].isdigit():
                    actual_layer = int(self.architecture[i])
                    encoder.add(Dense(actual_layer, input_dim=previous_layer,
                                    kernel_initializer=init))
                    previous_layer = actual_layer
                elif self.architecture[i] == 'l':
                    continue
                else:
                    raise(ValueError, 'Architecture with a wrong specification')

            decoder = Sequential()
            for i in reversed(range(1, len(self.architecture)-1)):
                if self.architecture[i] == 'd':
                    decoder.add(Dropout(0.5))
                elif self.architecture[i] == 's':
                    decoder.add(Activation('sigmoid'))
                elif self.architecture[i] == 'm':
                    decoder.add(Dense(input_size, kernel_initializer=init))
                    decoder.add(Activation('softmax'))
                elif self.architecture[i].isdigit():
                    actual_layer = int(self.architecture[i])
                    decoder.add(Dense(actual_layer, input_dim=previous_layer,
                                    kernel_initializer=init))
                    previous_layer = actual_layer
                elif self.architecture[i] == 'l':
                    continue
                else:
                    raise(ValueError, 'Architecture with a wrong specification')
            decoder.add(Dense(input_dim, input_dim=previous_layer,
                            kernel_initializer=init))
        else:
            raise(ValueError, 'Architecture with a wrong specification')

        model = Sequential()
        model.add(encoder)
        model.add(decoder)
        self.encoder = encoder
        self.decoder = decoder
        print(encoder.summary())
        print(decoder.summary())
        print(model.summary())

        if optimizer == 'sgd':
            optimizer = SGD(lr=lr, momentum=momentum, decay=decay,
                            nesterov=nesterov)

        loss = 'mean_squared_error'
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def transform(self, x):
        return self.encoder.predict(x, verbose=0)

    def inverse_transform(self, h):
        return self.decoder.predict(h, verbose=0)

    def score(self, x, pred):
        return mse(x, pred)

def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', '--architecture', type=str,
                        default='auto2l',
                        help='''Autoencoder architecture in the following form:
                        start always with auto, then add hidden layers
                        specifying the number of units and the activation
                        functions with a letter. The available letters are s
                        for sigmoid, l for linear, m for softmax, d for
                        dropout with fixed value of 0.5. Eg. 'auto20s10s2s'
                        will generate an autoencoder with 20 sigmoid units, 10
                        sigmoid units 2 sigmoid units, 2 sigmoid units, 10
                        sigmoid units 20 sigmoid units and input_size units.''')
    parser.add_argument('-d', '--dataset', type=str, default='iris',
                        help='''Dataset to use. Working options: iris,
                                breast_cancer, wine, digits''')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to run the classifier')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Seed for the data shuffle')
    parser.add_argument('-b', '--batchsize', type=int, default=1,
                        help='Batch size during training')
    parser.add_argument('-l', '--learning-rate', type=float, default=1.0,
                        help='Learning rate')

    return parser.parse_args()


def main(dataset, architecture, epochs, seed, batchsize, learning_rate):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = getattr(datasets, 'load_{}'.format(dataset))()
    data_tool = hs.plugins.sklearn.tools.dataset(data, shuffle=True,
                                                 epochs=epochs, seed=seed)
    data_stream = M.get_or_create_stream('dataset')

    model = MyKerasUnsupervised(architecture=architecture, lr=learning_rate)
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

    unsupervised_tool.execute(sources=[mini_batch_stream], sink=unsupervised_stream,
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
