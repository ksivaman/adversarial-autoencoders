# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the label smoothing defence in `LabelSmoothing`. It computes a vector of smooth labels from a
vector of hard labels.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class LabelSmoothing(Preprocessor):
    """
    Computes a vector of smooth labels from a vector of hard ones. The hard labels have to contain ones for the
    correct classes and zeros for all the others. The remaining probability mass between `max_value` and 1 is
    distributed uniformly between the incorrect classes for each instance.
    """
    params = ['max_value']

    def __init__(self, max_value=0.9, apply_fit=True, apply_predict=False):
        """
        Create an instance of label smoothing.

        :param max_value: Value to affect to correct label
        :type max_value: `float`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(LabelSmoothing, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(max_value=max_value)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Apply label smoothing.

        :param x: Input data, will not be modified by this method
        :type x: `np.ndarray`
        :param y: Original vector of label probabilities (one-vs-rest)
        :type y: `np.ndarray`
        :return: Unmodified input data and the vector of smooth probabilities as correct labels
        :rtype: `(np.ndarray, np.ndarray)`
        """

        min_value = (1 - self.max_value) / (y.shape[1] - 1)
        assert self.max_value >= min_value

        smooth_y = y.copy()
        smooth_y[smooth_y == 1.] = self.max_value
        smooth_y[smooth_y == 0.] = min_value
        return x, smooth_y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """No parameters to learn for this method; do nothing."""
        pass

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        Defense-specific parameters:
        :param max_value: Value to affect to correct label
        :type max_value: `float`
        """
        # Save attack-specific parameters
        super(LabelSmoothing, self).set_params(**kwargs)

        if self.max_value <= 0 or self.max_value > 1:
            raise ValueError("The maximum value for correct labels must be between 0 and 1.")

        return True
