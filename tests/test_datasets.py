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
from pytz import UTC
from sklearn.datasets import load_iris
import numpy as np

from hyperstream import TimeInterval
from .helpers import *


class TestDatasets(unittest.TestCase):
    def run(self, result=None):
        with resource_manager() as resource:
            self.hs = resource
            super(TestDatasets, self).run(result)

    def test_iris(self):
        M = self.hs.channel_manager.memory
        T = self.hs.plugins.sklearn.tools

        data = load_iris()
        epochs = 10
        seed = 42
        batchsize = 2

        data_tool = T.dataset(data, shuffle=True, epochs=epochs, seed=seed)
        data_stream = M.get_or_create_stream('dataset')

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

        key, value = mini_batch_stream.window().items()[0]

        assert(key == datetime(1970, 1, 1, 0, 2, tzinfo=UTC))

        expected_value = {'x_te': np.array([[ 5.6,  2.8,  4.9,  2. ],
                                           [ 7.3,  2.9,  6.3,  1.8]]),
                         'x_tr': np.array([[ 6. ,  2.2,  5. ,  1.5],
                                           [ 5. ,  2. ,  3.5,  1. ]]),
                         'y_te': np.array([[0, 0, 1], [0, 0, 1]]),
                         'y_tr': np.array([[0, 0, 1], [0, 1, 0]])}

        for e_key, e_value in expected_value.items():
            assert(e_key in value)
            np.testing.assert_equal(e_value, value[e_key])
