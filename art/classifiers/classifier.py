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
This module implements the abstract base class `Classifier` for all classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

import numpy as np

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Classifier(ABC):
    """
    Base class for all classifiers.
    """

    def __init__(self, channel_index, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Initialize a `Classifier` object.

        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param defences: Defence(s) to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        from art.defences.preprocessor import Preprocessor

        if clip_values is not None:
            if len(clip_values) != 2:
                raise ValueError('`clip_values` should be a tuple of 2 floats or arrays containing the allowed'
                                 'data range.')
            if np.array(clip_values[0] >= clip_values[1]).any():
                raise ValueError('Invalid `clip_values`: min >= max.')
        self._clip_values = clip_values

        self._channel_index = channel_index
        if isinstance(defences, Preprocessor):
            self.defences = [defences]
        else:
            self.defences = defences

        if len(preprocessing) != 2:
            raise ValueError('`preprocessing` should be a tuple of 2 floats with the substract and divide values for'
                             'the model inputs.')
        self.preprocessing = preprocessing

    @abc.abstractmethod
    def predict(self, x, logits=False, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator `gen` that yields batches as specified. Framework implementations can
        provide framework-specific versions of this function to speed-up computation.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import DataGenerator

        if not isinstance(generator, DataGenerator):
            raise ValueError('Expected instance of `DataGenerator` for `fit_generator`, got %s instead.'
                             % str(type(generator)))

        for _ in range(nb_epochs):
            x, y = generator.get_batch()

            # Apply preprocessing and defences
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

            # Fit for current batch
            self.fit(x_preprocessed, y_preprocessed, nb_epochs=1, batch_size=len(x), **kwargs)

    @property
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    @property
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        return self._input_shape

    @property
    def clip_values(self):
        """
        :return: Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        :rtype: `tuple`
        """
        return self._clip_values

    @property
    def channel_index(self):
        """
        :return: Index of the axis in data containing the color channels or features.
        :rtype: `int`
        """
        return self._channel_index

    @property
    def learning_phase(self):
        """
        Return the learning phase set by the user for the current classifier. Possible values are `True` for training,
        `False` for prediction and `None` if it has not been set through the library. In the latter case, the library
        does not do any explicit learning phase manipulation and the current value of the backend framework is used.
        If a value has been set by the user for this property, it will impact all following computations for
        model fitting, prediction and gradients.

        :return: Value of the learning phase.
        :rtype: `bool` or `None`
        """
        return self._learning_phase if hasattr(self, '_learning_phase') else None

    @abc.abstractmethod
    def class_gradient(self, x, label=None, logits=False, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError

    def _apply_preprocessing(self, x, y, fit):
        """
        Apply all preprocessing steps of the classifier on inputs `(x, y)`.

        :param x: Input data, where first dimension is the batch size.
        :type x: `np.ndarray`
        :param y: Labels for input data, where first dimension is the batch size.
        :type y: `np.ndarray`
        :param fit: `True` if the defences are applied during training.
        :return: Value of the data after applying the defences.
        :rtype: `np.ndarray`
        """
        x_preprocessed, y_preprocessed = self._apply_preprocessing_defences(x, y, fit=fit)
        x_preprocessed = self._apply_preprocessing_normalization(x_preprocessed)
        return x_preprocessed, y_preprocessed

    def _apply_preprocessing_gradient(self, x, grads):
        """
        Apply the backward pass through all preprocessing steps to gradients.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :type x: `np.ndarray`
        :param grads: Gradient value so far.
        :type grads: `np.ndarray`
        :param fit: `True` if the gradient is computed during training.
        :return: Value of the gradient.
        :rtype: `np.ndarray`
        """
        grads = self._apply_preprocessing_normalization_gradient(grads)
        grads = self._apply_preprocessing_defences_gradient(x, grads)
        return grads

    def _apply_preprocessing_defences(self, x, y, fit=False):
        """
        Apply the defences specified for the classifier in inputs `(x, y)`.

        :param x: Input data, where first dimension is the batch size.
        :type x: `np.ndarray`
        :param y: Labels for input data, where first dimension is the batch size.
        :type y: `np.ndarray`
        :param fit: `True` if the defences are applied during training.
        :return: Value of the data after applying the defences.
        :rtype: `np.ndarray`
        """
        if self.defences is not None:
            for defence in self.defences:
                if fit:
                    if defence.apply_fit:
                        x, y = defence(x, y)
                else:
                    if defence.apply_predict:
                        x, y = defence(x, y)

        return x, y

    def _apply_preprocessing_defences_gradient(self, x, grads, fit=False):
        """
        Apply the backward pass through the preprocessing defences.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :type x: `np.ndarray`
        :param grads: Gradient value so far.
        :type grads: `np.ndarray`
        :param fit: `True` if the gradient is computed during training.
        :return: Value of the gradient.
        :rtype: `np.ndarray`
        """
        if self.defences is not None:
            for defence in self.defences[::-1]:
                if fit:
                    if defence.apply_fit:
                        grads = defence.estimate_gradient(x, grads)
                else:
                    if defence.apply_predict:
                        grads = defence.estimate_gradient(x, grads)

        return grads

    def _apply_preprocessing_normalization(self, x):
        """
        Apply the data normalization steps specified for the classifier on `x`.

        :param x: Input data, where first dimension is the batch size.
        :type x: `np.ndarray`
        :return: Value of the preprocessed data.
        :rtype: `np.ndarray`
        """
        sub, div = self.preprocessing
        sub = np.asarray(sub, dtype=x.dtype)
        div = np.asarray(div, dtype=x.dtype)

        res = x - sub
        res = res / div

        return res

    def _apply_preprocessing_normalization_gradient(self, grads):
        """
        Apply the backward pass through the data normalization steps.

        :param grads: Gradient value so far.
        :type grads: `np.ndarray`
        :return: Value of the gradient.
        :rtype: `np.ndarray`
        """
        _, div = self.preprocessing
        div = np.asarray(div, dtype=grads.dtype)
        res = grads / div
        return res

    def __repr__(self):
        repr_ = "%s(channel_index=%r, clip_values=%r, defences=%r, preprocessing=%r)" \
                % (self.__module__ + '.' + self.__class__.__name__,
                   self.channel_index, self.clip_values, self.defences, self.preprocessing)

        return repr_
