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
This module implements the virtual adversarial attack. It was originally was used for virtual adversarial training.

Paper link:
    https://arxiv.org/abs/1507.00677
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art import NUMPY_DTYPE
from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class VirtualAdversarialMethod(Attack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677
    """
    attack_params = Attack.attack_params + ['eps', 'finite_diff', 'max_iter', 'batch_size']

    def __init__(self, classifier, max_iter=10, finite_diff=1e-6, eps=.1, batch_size=1):
        """
        Create a VirtualAdversarialMethod instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param eps: Attack step (max input variation).
        :type eps: `float`
        :param finite_diff: The finite difference parameter.
        :type finite_diff: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param batch_size: Batch size.
        :type batch_size: `int`
        """
        super(VirtualAdversarialMethod, self).__init__(classifier)
        kwargs = {'finite_diff': finite_diff,
                  'eps': eps,
                  'max_iter': max_iter,
                  'batch_size': batch_size}
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: An array with the original labels to be predicted.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.astype(NUMPY_DTYPE)
        preds = self.classifier.predict(x_adv, logits=False, batch_size=self.batch_size)

        # Pick a small scalar to avoid division by 0
        tol = 1e-10

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch = batch.reshape((batch.shape[0], -1))

            # Main algorithm for each batch
            var_d = np.random.randn(*batch.shape)

            # Main loop of the algorithm
            for _ in range(self.max_iter):
                var_d = self._normalize(var_d)
                preds_new = self.classifier.predict((batch + var_d).reshape((-1,) + self.classifier.input_shape),
                                                    logits=False)

                from scipy.stats import entropy
                kl_div1 = entropy(np.transpose(preds[batch_index_1:batch_index_2]), np.transpose(preds_new))

                var_d_new = np.zeros(var_d.shape)
                for current_index in range(var_d.shape[1]):
                    var_d[:, current_index] += self.finite_diff
                    preds_new = self.classifier.predict((batch + var_d).reshape((-1,) + self.classifier.input_shape),
                                                        logits=False)
                    kl_div2 = entropy(np.transpose(preds[batch_index_1:batch_index_2]), np.transpose(preds_new))
                    var_d_new[:, current_index] = (kl_div2 - kl_div1) / (self.finite_diff + tol)
                    var_d[:, current_index] -= self.finite_diff
                var_d = var_d_new

            # Apply perturbation and clip
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                x_adv[batch_index_1:batch_index_2] = \
                    np.clip(batch + self.eps * self._normalize(var_d), clip_min, clip_max) \
                    .reshape((-1,) + self.classifier.input_shape)
            else:
                x_adv[batch_index_1:batch_index_2] = (batch + self.eps * self._normalize(var_d)) \
                    .reshape((-1,) + self.classifier.input_shape)

        return x_adv

    @staticmethod
    def _normalize(x):
        """
        Apply L_2 batch normalization on `x`.

        :param x: The input array batch to normalize.
        :type x: `np.ndarray`
        :return: The normalized version of `x`.
        :rtype: `np.ndarray`
        """
        tol = 1e-10

        inverse = (np.sum(x ** 2, axis=1) + tol) ** -.5
        x = x * inverse[:, None]

        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param eps: Attack step (max input variation).
        :type eps: `float`
        :param finite_diff: The finite difference parameter.
        :type finite_diff: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(VirtualAdversarialMethod, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.eps <= 0:
            raise ValueError("The attack step must be positive.")

        if not isinstance(self.finite_diff, float) or self.finite_diff <= 0:
            raise ValueError("The finite difference parameter must be a positive float.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True
