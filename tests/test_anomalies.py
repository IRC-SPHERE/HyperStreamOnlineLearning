# The MIT License (MIT) # Copyright (c) 2014-2017 University of Bristol
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.

import unittest
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
import numpy as np

from hyperstream import TimeInterval
from .helpers import *

nan = float('nan')
true_means = [
  nan,   nan ,   nan,  0.92,  0.93,  0.57,  0.52,  0.91,  0.81,  0.41,  0.84,  0.53,
  0.77,  0.88,  0.95,  0.79,  0.58,  0.84,  0.69,  0.82,  0.63,  0.55,  0.99,  0.9 ,
  0.89,  0.65,  0.61,  0.83,  0.87,  0.86,  0.78,  0.75,  0.74,  0.66,  0.84,  0.92,
  0.81,  0.9 ,  0.99,  0.72,  0.78,  0.84,  0.6 ,  0.81,  0.62,  0.83,  0.6 ,  0.42,
  0.87,  0.6 ,  0.8 ,  0.89,  0.77,  0.88,  0.72,  0.75,  0.81,  0.42,  0.81,  0.67,
  0.87,  0.89,  0.8 ,  0.56,  0.67,  0.8 ,  0.74,  0.8 ,  0.71,  0.66,  0.75,  0.82,
  0.73,  0.82,  0.87,  0.98,  0.75,  0.95,  0.71,  0.73,  0.72,  0.55,  0.83,  0.83,
  0.33,  0.81,  0.48,  0.77,  0.88,  0.93,  0.78,  0.6 ,  0.83,  0.69,  0.76,  0.64,
  0.53,  0.98,  0.87,  0.82,  0.63,  0.55,  0.85,  0.81,  0.81,  0.74,  0.78,  0.71,
  0.66,  0.82,  0.91,  0.8 ,  0.89,  0.98,  0.73,  0.78,  0.84,  0.59,  0.85,  0.61,
  0.83,  0.58,  0.42,  0.85,  0.57,  0.79,  0.91,  0.75,  0.88,  0.71,  0.73,  0.8 ,
  0.42,  0.82,  0.68,  0.87,  0.88,  0.8 ,  0.56,  0.67,  0.8 ,  0.75,  0.81,  0.71,
  0.67,  0.76,  0.82,  0.73,  0.82,  0.87,  0.98,  0.75,  0.95,  0.71,  0.74,  0.73,
  0.56,  0.83,  0.83,  0.33,  0.81,  0.48,  0.77,  0.88,  0.93,  0.78,  0.61,  0.83,
  0.69,  0.76,  0.64,  0.53,  0.97,  0.86,  0.82,  0.62,  0.54,  0.85,  0.8 ,  0.8 ,
  0.74,  0.78,  0.71,  0.66,  0.82,  0.9 ,  0.8 ,  0.89,  0.98,  0.74,  0.78,  0.85,
  0.59,  0.86,  0.61,  0.83,  0.58,  0.42,  0.85,  0.56,  0.79,  0.91,  0.75,  0.88,
  0.71,  0.73,  0.8 ,  0.42,  0.82,  0.68,  0.87,  0.87,  0.8 ,  0.56,  0.68,  0.8 ,
  0.75,  0.81,  0.72,  0.68,  0.76,  0.82,  0.73,  0.82,  0.87,  0.98,  0.75,  0.95,
  0.71,  0.74,  0.73,  0.56,  0.83,  0.84,  0.33,  0.81,  0.48,  0.77,  0.88,  0.93,
  0.78,  0.61,  0.83,  0.69,  0.75,  0.64,  0.53,  0.97,  0.86,  0.81,  0.62,  0.54,
  0.85,  0.8 ,  0.8 ,  0.73,  0.78,  0.71,  0.66,  0.82,  0.9 ,  0.8 ,  0.89,  0.98,
  0.74,  0.78,  0.85,  0.59,  0.86,  0.61,  0.83,  0.58,  0.42,  0.85,  0.56,  0.79,
  0.91,  0.75,  0.88,  0.71,  0.73,  0.79,  0.42,  0.82,  0.68,  0.87,  0.87,  0.8 ,
  0.56,  0.68,  0.8 ,  0.75,  0.81,  0.72,  0.68,  0.76,  0.82,  0.73,  0.82,  0.87,
  0.98,  0.75,  0.95,  0.71,  0.74,  0.73,  0.56,  0.82,  0.84,  0.33,  0.81,  0.48,
  0.78,  0.88,  0.93,  0.78,  0.61,  0.83,  0.69,  0.75,  0.64,  0.53,  0.97,  0.86,
  0.81,  0.62,  0.54,  0.85,  0.8 ,  0.8 ,  0.73,  0.79,  0.71,  0.66,  0.82,  0.9 ,
  0.8 ,  0.89,  0.98,  0.74,  0.78,  0.85,  0.59,  0.87,  0.61,  0.83,  0.58,  0.42,
  0.85,  0.56,  0.79,  0.91,  0.75,  0.88,  0.71,  0.73,  0.79,  0.42,  0.82,  0.68,
  0.87,  0.87,  0.8 ,  0.56,  0.68,  0.8 ,  0.75,  0.82,  0.72,  0.68,  0.76,  0.82,
  0.73,  0.82,  0.87]


class TestAnomalies(unittest.TestCase):
    def run(self, result=None):
        with resource_manager() as resource:
            self.hs = resource
            super(TestAnomalies, self).run(result)

    def test_data_generators(self):
        M = self.hs.channel_manager.memory
        T = self.hs.plugins.sklearn.tools

        data = load_iris()
        epochs = 10
        seed = 42
        batchsize = 2

        data_tool = T.dataset(data, shuffle=True, epochs=epochs, seed=seed)
        data_stream = M.get_or_create_stream('dataset')
        model = 'Gaussian'

        anomaly_detector_tool = T.anomaly_detector(model)
        anomaly_detector_stream = M.get_or_create_stream('anomaly_detector')

        now = datetime.utcnow()
        now = (now - timedelta(hours=1))
        before = datetime.utcfromtimestamp(0)
        ti = TimeInterval(before, now)

        data_tool.execute(sources=[], sink=data_stream, interval=ti)

        print("Example of a data stream")
        key, value = next(iter(data_stream.window()))
        print('[%s]: %s' % (key, value))

        mini_batch_tool = T.minibatch(batchsize=batchsize)
        mini_batch_stream = M.get_or_create_stream('mini_batch')
        mini_batch_tool.execute(sources=[data_stream], sink=mini_batch_stream,
                                interval=ti)

        anomaly_detector_tool.execute(sources=[mini_batch_stream],
                                      sink=anomaly_detector_stream, interval=ti)

        probas = []
        for key, value in anomaly_detector_stream.window():
            probas.append(value['proba'])

        # The data is repeated the number of epochs. This makes the mini-batches to
        # cycle and contain data from the beginning and end of the dataset. This
        # makes possible that the number of scores is not divisible by epochs.
        probas = np.array(probas)
        print(probas.shape)
        means = np.array([np.nanmean(aux) for aux in probas])
        np.testing.assert_almost_equal(true_means, means, decimal=2)
        print(means.shape)
        print("Test probabilities per minibatch (cyclic)")
        print(means.round(decimals=2))
