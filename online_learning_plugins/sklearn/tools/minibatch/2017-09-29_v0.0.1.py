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
import itertools

from hyperstream import Tool, StreamInstance
from hyperstream.utils import check_input_stream_count

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
        """
        Tool that aggregates several streams together and creates batches

        Parameters
        ==========
        batchsize: integer
            Number of streams to be joint
        """
        super(Minibatch, self).__init__(batchsize=batchsize)

    @check_input_stream_count(1)
    def _execute(self, sources, alignment_stream, interval):
        s0 = sources[0].window(interval, force_calculation=True).items()

        for group in grouper(self.batchsize, s0):
            values = {}
            for stream in group:
                for key, value in stream.value.items():
                    if key in values:
                        values[key].append(value)
                    else:
                        values[key] = [value]
            for key in values.keys():
                values[key] = np.concatenate(values[key])
            yield StreamInstance(group[-1].timestamp, values)
