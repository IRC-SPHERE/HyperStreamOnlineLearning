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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

from datetime import datetime, timedelta
from dateutil.parser import parse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np
from pytz import UTC

class Dataset(Tool):
    def __init__(self, dataset, shuffle=True, epochs=1):
        super(Dataset, self).__init__(dataset=dataset, shuffle=shuffle,
                                      epochs=epochs)

    @check_input_stream_count(0)
    def _execute(self, sources, alignment_stream, interval):

        x = self.dataset.data
        y = self.dataset.target
        # Binarize data
        classes = np.unique(y)
        y = label_binarize(y, classes)

        X_tr, X_te, Y_tr, Y_te = train_test_split(x, y, shuffle=self.shuffle,
                                                  train_size=0.5, stratify=y)

        j = 0
        start_dt = datetime.utcfromtimestamp(0).replace(tzinfo=UTC)
        for i in range(self.epochs):
            for x_tr, y_tr in zip(X_tr, Y_tr):
                x_te, y_te = X_te[j%len(X_te)], Y_te[j%len(Y_te)]
                j += 1
                dt = (start_dt + timedelta(minutes=j)).replace(tzinfo=UTC)
                yield StreamInstance(dt, dict(x_tr=x_tr.reshape(1,-1),
                                              x_te=x_te.reshape(1,-1),
                                              y_tr=y_tr.reshape(1,-1),
                                              y_te=y_te.reshape(1,-1)))
