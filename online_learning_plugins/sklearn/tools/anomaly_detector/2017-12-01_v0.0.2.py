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
from online_learning_plugins.sklearn.utils import GaussianEstimator, BackgroundCheck


class AnomalyDetector(Tool):
    def __init__(self, model, **fit_arguments):
        """
        It requires an unsupervised online-learning model.

        Parameters
        ==========
        model: object with the following functions
            partial_fit or fit(x, **kwargs):
                trains the model with x
            predict_proba or predict or likelihood(x):
                returns a prediction for x

        fit_arguments: dictionary (Not implemented yet)
            Dictionary that will be passed to the fit function as **kwargs
        """
        super(AnomalyDetector, self).__init__(model=model,
                                              fit_arguments=fit_arguments)
        if model == 'Gaussian':
            self.model = BackgroundCheck(GaussianEstimator())
        else:
            raise (ValueError('Unknown model {}'.format(model)))

    @check_input_stream_count(1)
    def _execute(self, sources, alignment_stream, interval):
        s0 = sources[0].window(interval, force_calculation=True).items()

        first = True
        for dt, value in s0:
            # TODO consider the case with training or test
            x_tr = value['x_tr']
            x_te = value['x_te']
            if len(x_tr.shape) == 1:
                raise(ValueError('Anomaly detector needs more than one sample'))

            if first:
                if hasattr(self.model, 'partial_fit'):
                    self.fit = self.model.partial_fit
                else:
                    self.fit = self.model.fit

                if hasattr(self.model, 'predict_proba'):
                    self.predict_proba = self.model.predict_proba
                elif hasattr(self.model, 'predict'):
                    self.predict_proba = self.model.predict
                elif hasattr(self.model, 'likelihood'):
                    self.predict_proba = self.model.likelihood

                first = False

            self.fit(x_tr, **self.fit_arguments)
            proba = self.predict_proba(x_te)
            yield StreamInstance(dt, dict(proba=proba))
