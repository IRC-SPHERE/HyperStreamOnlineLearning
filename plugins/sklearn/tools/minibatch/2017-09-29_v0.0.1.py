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

import itertools
import numpy as np

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

class Minibatch(Tool):
    def __init__(self, batchsize, **fit_arguments):
        super(Minibatch, self).__init__(batchsize=batchsize)

    @check_input_stream_count(1)
    def _execute(self, sources, alignment_stream, interval):
        s0 = sources[0].window(interval, force_calculation=True).items()

        i = 1
        for batch in grouper(self.batchsize, s0):
            i += 1
            dt = batch[0][0]
            # Try to reduce these 5 lines into one?
            x_te = np.concatenate([b[1]['x_te'] for b in batch])
            x_tr = np.concatenate([b[1]['x_tr'] for b in batch])
            y_te = np.concatenate([b[1]['y_te'] for b in batch])
            y_tr = np.concatenate([b[1]['y_tr'] for b in batch])
            value = dict(x_te=x_te, x_tr=x_tr, y_te=y_te, y_tr=y_tr)
            yield StreamInstance(dt, value)
            # Try to reduce these 5 lines into one?
