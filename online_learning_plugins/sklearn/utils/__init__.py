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

import numpy as np


class BackgroundCheck(object):
    def __init__(self, model):
        self.model = model

    def fit(self, x):
        self.model.fit(x)

    def prob_foreground(self, x):
        l = self.model.likelihood(x)
        l_max = self.model.max
        return np.true_divide(l, l_max)

    def prob_background(self, x):
        return 1 - self.prob_foreground(x)

    def predict_proba(self, x):
        return self.prob_background(x)


class GaussianEstimator(object):
    def __init__(self):
        self.mu = None
        self.cov = None
        self.N = 0
        self.k = None

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
        inverse = np.linalg.pinv(self.cov)
        exp = np.array([np.inner(np.inner(a, inverse.T), a) for a in x_mu])
        return - 0.5 * (
            np.log(np.linalg.det(self.cov))
            + exp + self.k * np.log(2 * np.pi)
        )

    @property
    def max(self):
        return self.likelihood(self.mu.reshape(1, -1))
