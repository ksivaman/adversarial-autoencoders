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
This module implements the base class `DetectorClassifier` for classifier and detector combinations.

Paper link:
    https://arxiv.org/abs/1705.07263
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class DetectorClassifier(Classifier):
    """
    This class implements a Classifier extension that wraps a classifier and a detector.
    More details in https://arxiv.org/abs/1705.07263
    """

    def __init__(self, classifier, detector, defences=None, preprocessing=(0, 1)):
        """
        Initialization for the DetectorClassifier.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param detector: A trained detector applied for the binary classification.
        :type detector: `art.detection.detector.Detector`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(DetectorClassifier, self).__init__(clip_values=classifier.clip_values, preprocessing=preprocessing,
                                                 channel_index=classifier.channel_index, defences=defences)

        self.classifier = classifier
        self.detector = detector
        self._nb_classes = classifier.nb_classes + 1
        self._input_shape = classifier.input_shape

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
        # Apply preprocessing
        x_defences, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Compute the prediction logits
        classifier_logits = self.classifier.predict(x=x_defences, logits=True, batch_size=batch_size)
        detector_logits = self.detector.predict(x=x_defences, logits=True, batch_size=batch_size)
        detector_logits = (np.reshape(detector_logits, [-1]) + 1) * np.max(classifier_logits, axis=1)
        detector_logits = np.reshape(detector_logits, [-1, 1])
        combined_logits = np.concatenate([classifier_logits, detector_logits], axis=1)

        if logits:
            result = combined_logits
        else:
            result = np.exp(combined_logits) / np.sum(np.exp(combined_logits), axis=1)[:, None]

        return result

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
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
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

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
        if not ((label is None) or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self._nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % label)

        # Apply preprocessing
        x_defences, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Compute the gradient and return
        if logits:
            if label is None:
                combined_grads = self._compute_combined_grads(x, label=None)

            elif isinstance(label, (int, np.int)):
                if label < self._nb_classes - 1:
                    # Compute and return from the classifier gradients
                    combined_grads = self.classifier.class_gradient(x=x_defences, label=label, logits=True)

                else:
                    # First compute the classifier gradients
                    classifier_grads = self.classifier.class_gradient(x=x_defences, label=None, logits=True)

                    # Then compute the detector gradients
                    detector_grads = self.detector.class_gradient(x=x_defences, label=0, logits=True)

                    # Chain the detector gradients for the first component
                    classifier_logits = self.classifier.predict(x=x_defences, logits=True)
                    maxind_classifier_logits = np.argmax(classifier_logits, axis=1)
                    max_classifier_logits = classifier_logits[np.arange(x_defences.shape[0]), maxind_classifier_logits]
                    first_detector_grads = max_classifier_logits[:, None, None, None, None] * detector_grads

                    # Chain the detector gradients for the second component
                    max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_logits]
                    detector_logits = self.detector.predict(x=x_defences, logits=True)
                    second_detector_grads = max_classifier_grads * (detector_logits + 1)[:, None, None]
                    second_detector_grads = second_detector_grads[None, ...]
                    second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

                    # Update detector gradients
                    combined_grads = first_detector_grads + second_detector_grads

            else:
                # Compute indexes for classifier labels and detector labels
                classifier_idx = np.where(label < self._nb_classes - 1)
                detector_idx = np.where(label == self._nb_classes - 1)

                # Initialize the combined gradients
                combined_grads = np.zeros(shape=(x_defences.shape[0], 1, x_defences.shape[1], x_defences.shape[2],
                                                 x_defences.shape[3]))

                # First compute the classifier gradients for classifier_idx
                if classifier_idx:
                    combined_grads[classifier_idx] = self.classifier.class_gradient(x=x_defences[classifier_idx],
                                                                                    label=label[classifier_idx],
                                                                                    logits=True)

                # Then compute the detector gradients for detector_idx
                if detector_idx:
                    # First compute the classifier gradients for detector_idx
                    classifier_grads = self.classifier.class_gradient(x=x_defences[detector_idx], label=None,
                                                                      logits=True)

                    # Then compute the detector gradients for detector_idx
                    detector_grads = self.detector.class_gradient(x=x_defences[detector_idx], label=0, logits=True)

                    # Chain the detector gradients for the first component
                    classifier_logits = self.classifier.predict(x=x_defences[detector_idx], logits=True)
                    maxind_classifier_logits = np.argmax(classifier_logits, axis=1)
                    max_classifier_logits = classifier_logits[np.arange(len(detector_idx)), maxind_classifier_logits]
                    first_detector_grads = max_classifier_logits[:, None, None, None, None] * detector_grads

                    # Chain the detector gradients for the second component
                    max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_logits]
                    detector_logits = self.detector.predict(x=x_defences[detector_idx], logits=True)
                    second_detector_grads = max_classifier_grads * (detector_logits + 1)[:, None, None]
                    second_detector_grads = second_detector_grads[None, ...]
                    second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

                    # Update detector gradients
                    detector_grads = first_detector_grads + second_detector_grads

                    # Reassign the combined gradients
                    combined_grads[detector_idx] = detector_grads

        else:
            # First compute the combined logits
            combined_logits = self.predict(x=x_defences, logits=True)
            sum_exp_logits = np.sum(np.exp(combined_logits), axis=1)
            combined_logits_grads = self._compute_combined_grads(x_defences, label=None)

            # Now compute the softmax gradients for each of the three cases
            grads = []
            if label is None:
                for i in range(self._nb_classes):
                    si_grads = 0
                    for j in range(self._nb_classes):
                        c_var = sum_exp_logits - np.exp(combined_logits[:, j])

                        if j == i:
                            si_lj = c_var * np.power(c_var + np.exp(combined_logits[:, i]), -2) * np.exp(
                                combined_logits[:, i])
                        else:
                            si_lj = -np.exp(combined_logits[:, i]) * np.power(c_var + np.exp(combined_logits[:, j]),
                                                                              -2) * np.exp(combined_logits[:, j])
                        si_grads += si_lj[:, None, None, None] * combined_logits_grads[:, j]

                    grads.append(si_grads)

                combined_grads = np.swapaxes(np.array(grads), 0, 1)

            elif isinstance(label, (int, np.int)):
                si_grads = 0
                for j in range(self._nb_classes):
                    c_var = sum_exp_logits - np.exp(combined_logits[:, j])
                    if j == label:
                        si_lj = c_var * np.power(c_var + np.exp(combined_logits[:, label]), -2) * \
                                np.exp(combined_logits[:, label])
                    else:
                        si_lj = -np.exp(combined_logits[:, label]) * np.power(c_var + np.exp(combined_logits[:, j]),
                                                                              -2) * np.exp(combined_logits[:, j])
                    si_grads += si_lj[:, None, None, None] * combined_logits_grads[:, j]

                grads.append(si_grads)
                combined_grads = np.swapaxes(np.array(grads), 0, 1)

            else:
                unique_label = list(np.unique(label))
                for i in unique_label:
                    si_grads = 0
                    for j in range(self._nb_classes):
                        c_var = sum_exp_logits - np.exp(combined_logits[:, j])
                        if j == i:
                            si_lj = c_var * np.power(c_var + np.exp(combined_logits[:, i]), -2) * np.exp(
                                combined_logits[:, i])
                        else:
                            si_lj = -np.exp(combined_logits[:, i]) * np.power(c_var + np.exp(combined_logits[:, j]),
                                                                              -2) * np.exp(combined_logits[:, j])
                        si_grads += si_lj[:, None, None, None] * combined_logits_grads[:, j]

                    grads.append(si_grads)

                grads = np.swapaxes(np.array(grads), 0, 1)
                lst = [unique_label.index(i) for i in label]
                grads = grads[np.arange(len(grads)), lst]
                combined_grads = np.expand_dims(grads, axis=1)

        combined_grads = self._apply_preprocessing_gradient(x, combined_grads)

        return combined_grads

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
        Return the hidden layers in the model, if applicable. This function is not supported for the
        Classifier and Detector wrapper.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        raise NotImplementedError

    def get_activations(self, x, layer, batch_size=128):
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

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self.classifier.set_learning_phase(train=train)
            self.detector.set_learning_phase(train=train)

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
        self.classifier.save(filename=filename + "_classifier", path=path)
        self.detector.save(filename=filename + "_detector", path=path)

    def __repr__(self):
        repr_ = "%s(classifier=%r, detector=%r, defences=%r, preprocessing=%r)" \
                % (self.__module__ + '.' + self.__class__.__name__,
                   self.classifier, self.detector, self.defences, self.preprocessing)

        return repr_

    def _compute_combined_grads(self, x, label=None):

        # Compute the classifier gradients
        classifier_grads = self.classifier.class_gradient(x=x, label=label, logits=True)

        # Then compute the detector gradients
        detector_grads = self.detector.class_gradient(x=x, label=label, logits=True)

        # Chain the detector gradients for the first component
        classifier_logits = self.classifier.predict(x=x, logits=True)
        maxind_classifier_logits = np.argmax(classifier_logits, axis=1)
        max_classifier_logits = classifier_logits[np.arange(classifier_logits.shape[0]), maxind_classifier_logits]
        first_detector_grads = max_classifier_logits[:, None, None, None, None] * detector_grads

        # Chain the detector gradients for the second component
        max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_logits]
        detector_logits = self.detector.predict(x=x, logits=True)
        second_detector_grads = max_classifier_grads * (detector_logits + 1)[:, None, None]
        second_detector_grads = second_detector_grads[None, ...]
        second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

        # Update detector gradients
        detector_grads = first_detector_grads + second_detector_grads

        # Combine the gradients
        combined_logits_grads = np.concatenate([classifier_grads, detector_grads], axis=1)
        return combined_logits_grads
