# The MIT License (MIT)
# Copyright (c) 2014-2017 University of Bristol
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

from hyperstream import Tool, StreamInstance
from hyperstream.utils import check_input_stream_count

import numpy as np


class Unsupervised(Tool):
    """Trains and generates predictions with a machine learning model

    Tool that expects a stream of training and testing data and an online
    learning model and outputs the predictions of the model while at the same
    time trains it.
    """
    def __init__(self, model, **fit_arguments):
        """
        It requires an online-learning model.

        Parameters
        ==========
        model: object with the following functions
            partial_fit or fit(x, **kwargs):
                trains the model with x and y
            reconstruct(x):
                returns a prediction for x
            score(x):
                returns error of reconstructing x

        fit_arguments: dictionary (Not implemented yet)
            Dictionary that will be passed to the fit function as **kwargs
        """
        super(Unsupervised, self).__init__(model=model,
                                           fit_arguments=fit_arguments)

    @check_input_stream_count(1)
    def _execute(self, sources, alignment_stream, interval):
        """
        It expects at least one source of streams, each streams with a
        dictionary with training and test data in the form:
        x_tr: array of float
            Training values for the given data stream
        x_te: array of float
            Test values for the given data stream
        """
        s0 = sources[0].window(interval, force_calculation=True).items()

        first = True
        for dt, value in s0:
            x_tr = value['x_tr']
            x_te = value['x_te']
            if len(x_tr.shape) == 1:
                x_tr = x_tr.reshape(1, -1)
                x_te = x_te.reshape(1, -1)

            if first:
                if hasattr(self.model, 'partial_fit'):
                    self.fit = self.model.partial_fit
                else:
                    self.fit = self.model.fit

                self.fit(x_tr)

                first = False
            else:
                self.fit(x_tr)

            h_te = self.model.transform(x_te)
            pred_te = self.model.inverse_transform(h_te)

            score = self.model.score(x_te, pred_te)

            yield StreamInstance(dt, dict(score=score, reduced_x=h_te))
