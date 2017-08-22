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
from sklearn.multiclass import OneVsRestClassifier

from dateutil.parser import parse
import numpy as np

class Classifier(Tool):
    def __init__(self, model, **fit_arguments):
        super(Classifier, self).__init__(model=model,
                                        fit_arguments=fit_arguments)

    @check_input_stream_count(1)
    def _execute(self, sources, alignment_stream, interval):
        s0 = sources[0].window(interval, force_calculation=True).items()

        self.classifier = OneVsRestClassifier(self.model)

        first = True
        for dt, value in s0:
            x_tr = value['x_tr']
            x_te = value['x_te']
            if len(x_tr.shape) == 1:
                x_tr = x_tr.reshape(1,-1)
                x_te = x_val.reshape(1,-1)
            y_tr = np.argmax(value['y_tr']).reshape(1,-1)
            y_te = np.argmax(value['y_te']).reshape(1,-1)

            if first:
                self.classes = range(value['y_tr'].shape[1])
                self.classifier.partial_fit(x_tr, y_tr, self.classes)
                first = False
            else:
                self.classifier.partial_fit(x_tr, y_tr)
            proba_tr = self.classifier.predict_proba(x_tr)
            score = self.classifier.score(x_te, y_te)
            yield StreamInstance(dt, dict(proba=proba_tr, score=score))
