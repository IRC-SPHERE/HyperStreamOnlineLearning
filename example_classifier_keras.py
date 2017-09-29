import numpy as np
import argparse

from datetime import datetime, timedelta

from sklearn import datasets, linear_model
from sklearn.preprocessing import label_binarize

from hyperstream import HyperStream, TimeInterval
from hyperstream.utils import UTC

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

class MyKerasClassifier(object):
    """This is a modification of the KerasClassifier in order to keep the
    labels with the original values.

    Implementation of the scikit-learn classifier API for Keras.
    """
    #def __init__(self, optimizer='rmsprop', init='glorot_uniform', lr=1,
    #             momentum=0.0, decay=0.0, nesterov=False , architecture='lr'):
    #    return super(KerasClassifier, self).__init__()
    #
    def __init__(self, architecture='lr', lr=1):
        self.model = None
        self.architecture = architecture
        self.lr = lr

    def fit(self, x, y, classes=None, **kwargs):
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
        if classes is not None:
            self.classes = classes

        if y.shape[1] <= 1:
            y = label_binarize(y, self.classes)
        if self.model is None:
            self.model = self.create_model(input_dim=x.shape[1],
                                           output_size=y.shape[1],
                                           architecture=self.architecture,
                                           lr=self.lr)
        return self.model.fit(x, y, batch_size=x.shape[0], epochs=1, verbose=0)

    def create_model(self, input_dim=1, output_size=1, optimizer='rmsprop',
                     init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                     nesterov=False , architecture='lr'):
        """
        Parameters
        ----------
        architecture: string: lr, mlp100, mlp100d, mlp100d100d
        """

        model = Sequential()

        previous_layer = input_dim
        if architecture == 'lr':
            model.add(Dense(output_size, input_shape=(input_dim,),
                            kernel_initializer=init,
                            activation='softmax'))
        elif architecture.startswith('mlp'):
            architecture = architecture[3:]
            while len(architecture) > 0:
                if architecture[0] == 'd':
                    model.add(Dropout(0.5))
                    architecture = architecture[1:]
                elif architecture[0] == 's':
                    model.add(Activation('sigmoid'))
                    architecture = architecture[1:]
                elif architecture[0] == 'm':
                    model.add(Dense(output_size, kernel_initializer=init))
                    model.add(Activation('softmax'))
                    architecture = architecture[1:]
                elif architecture[0].isdigit():
                    i = 1
                    while len(architecture) > i and architecture[i].isdigit():
                        i += 1
                    actual_layer = int(architecture[:i])
                    model.add(Dense(actual_layer, input_dim=previous_layer,
                                    kernel_initializer=init))
                    architecture = architecture[i:]
                    previous_layer = actual_layer
                else:
                    raise(ValueError, 'Architecture with a wrong specification')
        else:
            raise(ValueError, 'Architecture with a wrong specification')

        if optimizer == 'sgd':
            optimizer = SGD(lr=lr, momentum=momentum, decay=decay,
                            nesterov=nesterov)

        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=optimizer,
                      metrics=['acc'])
        return model

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def score(self, x, y):
        pred = self.model.predict_classes(x, verbose=0).reshape(-1,1)
        return np.mean(pred == y)

def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--classifier', type=str,
                        default='lr',
                        help='''Classifier to use. Working options: lr for a
                        Logistic Regression. To specify Multilayer Perceptron
                        use the following convention: mlp100dm for a Multilayer
                        Perceptron with 100 hidden units, dropbout at 0.5 and
                        SoftMax activation, mlp30ds40m for an MLP with 30
                        hidden units with Dropout of 0.5, Sigmoid activation,
                        40 hidden units and Softmax activation''')
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


def main(dataset, classifier, epochs, seed, batchsize, learning_rate):
    hs = HyperStream(loglevel=30)
    print(hs)
    print([p.channel_id_prefix for p in hs.config.plugins])

    M = hs.channel_manager.memory

    data = getattr(datasets, 'load_{}'.format(dataset))()
    data_tool = hs.plugins.sklearn.tools.dataset(data, shuffle=True,
                                                 epochs=epochs, seed=seed)
    data_stream = M.get_or_create_stream('dataset')

    model = MyKerasClassifier(architecture=classifier, lr=learning_rate)
    classifier_tool = hs.plugins.sklearn.tools.classifier(model)
    classifier_stream = M.get_or_create_stream('classifier')

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

    classifier_tool.execute(sources=[mini_batch_stream], sink=classifier_stream,
                            interval=ti)

    scores = []
    for key, value in classifier_stream.window():
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
