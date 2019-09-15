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
This module implements the classifier `TFClassifier` for Tensorflow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random

import numpy as np
import six

from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """

    def __init__(self, input_ph, logits, output_ph=None, train=None, loss=None, learning=None, sess=None,
                 channel_index=3, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Initialization specific to Tensorflow models implementation.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param output_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        import tensorflow as tf

        super(TFClassifier, self).__init__(clip_values=clip_values, channel_index=channel_index, defences=defences,
                                           preprocessing=preprocessing)
        self._nb_classes = int(logits.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._logits = logits
        self._output_ph = output_ph
        self._train = train
        self._loss = loss
        self._learning = learning
        self._feed_dict = {}

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the internal layers
        self._layer_names = self._get_layers()

        # Must be set here for the softmax output
        self._probs = tf.nn.softmax(logits)

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

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
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Create feed_dict
            feed_dict = {self._input_ph: x_preprocessed[begin:end]}
            feed_dict.update(self._feed_dict)

            # Run prediction
            if logits:
                results[begin:end] = self._sess.run(self._logits, feed_dict=feed_dict)
            else:
                results[begin:end] = self._sess.run(self._probs, feed_dict=feed_dict)

        return results

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
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        # Check if train and output_ph available
        if self._train is None or self._output_ph is None:
            raise ValueError("Need the training objective and the output placeholder to train the model.")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = x_preprocessed[ind[m * batch_size:(m + 1) * batch_size]]
                o_batch = y_preprocessed[ind[m * batch_size:(m + 1) * batch_size]]

                # Create feed_dict
                feed_dict = {self._input_ph: i_batch, self._output_ph: o_batch}
                feed_dict.update(self._feed_dict)

                # Run train step
                self._sess.run(self._train, feed_dict=feed_dict)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in TensorFlow, it will.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import TFDataGenerator

        # Train directly in Tensorflow
        if isinstance(generator, TFDataGenerator) and not (hasattr(
                self, 'label_smooth') or hasattr(self, 'feature_squeeze')):
            for _ in range(nb_epochs):
                for _ in range(int(generator.size / generator.batch_size)):
                    i_batch, o_batch = generator.get_batch()

                    # Create feed_dict
                    feed_dict = {self._input_ph: i_batch, self._output_ph: o_batch}
                    feed_dict.update(self._feed_dict)

                    # Run train step
                    self._sess.run(self._train, feed_dict=feed_dict)
            super(TFClassifier, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

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
        # Check value of label for computing gradients
        if not (label is None or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self._nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % label)

        self._init_class_grads(label=label, logits=logits)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create feed_dict
        feed_dict = {self._input_ph: x_preprocessed}
        feed_dict.update(self._feed_dict)

        # Compute the gradient and return
        if label is None:
            # Compute the gradients w.r.t. all classes
            if logits:
                grads = self._sess.run(self._logit_class_grads, feed_dict=feed_dict)
            else:
                grads = self._sess.run(self._class_grads, feed_dict=feed_dict)

            grads = np.swapaxes(np.array(grads), 0, 1)
        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            if logits:
                grads = self._sess.run(self._logit_class_grads[label], feed_dict=feed_dict)
            else:
                grads = self._sess.run(self._class_grads[label], feed_dict=feed_dict)

            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)
        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            if logits:
                grads = self._sess.run([self._logit_class_grads[l] for l in unique_label], feed_dict=feed_dict)
            else:
                grads = self._sess.run([self._class_grads[l] for l in unique_label], feed_dict=feed_dict)

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

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
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check if loss available
        if not hasattr(self, '_loss_grads') or self._loss_grads is None or self._output_ph is None:
            raise ValueError("Need the loss function and the labels placeholder to compute the loss gradient.")

        # Create feed_dict
        feed_dict = {self._input_ph: x_preprocessed, self._output_ph: y_preprocessed}
        feed_dict.update(self._feed_dict)

        # Compute gradients
        grads = self._sess.run(self._loss_grads, feed_dict=feed_dict)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x_preprocessed.shape

        return grads

    def _init_class_grads(self, label=None, logits=False):
        import tensorflow as tf

        if logits:
            if not hasattr(self, '_logit_class_grads'):
                self._logit_class_grads = [None for _ in range(self.nb_classes)]
        else:
            if not hasattr(self, '_class_grads'):
                self._class_grads = [None for _ in range(self.nb_classes)]

        # Construct the class gradients graph
        if label is None:
            if logits:
                if None in self._logit_class_grads:
                    self._logit_class_grads = [tf.gradients(self._logits[:, i], self._input_ph)[0]
                                               if self._logit_class_grads[i] is None else self._logit_class_grads[i]
                                               for i in range(self._nb_classes)]
            else:
                if None in self._class_grads:
                    self._class_grads = [tf.gradients(self._probs[:, i], self._input_ph)[0]
                                         if self._class_grads[i] is None else self._class_grads[i]
                                         for i in range(self._nb_classes)]

        elif isinstance(label, int):
            if logits:
                if self._logit_class_grads[label] is None:
                    self._logit_class_grads[label] = tf.gradients(self._logits[:, label], self._input_ph)[0]
            else:
                if self._class_grads[label] is None:
                    self._class_grads[label] = tf.gradients(self._probs[:, label], self._input_ph)[0]

        else:
            if logits:
                for unique_label in np.unique(label):
                    if self._logit_class_grads[unique_label] is None:
                        self._logit_class_grads[unique_label] = \
                            tf.gradients(self._logits[:, unique_label], self._input_ph)[0]
            else:
                for unique_label in np.unique(label):
                    if self._class_grads[unique_label] is None:
                        self._class_grads[unique_label] = tf.gradients(self._probs[:, unique_label], self._input_ph)[0]

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        # Get the list of operators and heuristically filter them
        tmp_list = []
        ops = graph.get_operations()

        # pylint: disable=R1702
        for op in ops:
            if op.values():
                if op.values()[0].get_shape() is not None:
                    if op.values()[0].get_shape().ndims is not None:
                        if len(op.values()[0].get_shape().as_list()) > 1:
                            if op.values()[0].get_shape().as_list()[0] is None:
                                if op.values()[0].get_shape().as_list()[1] is not None:
                                    if not op.values()[0].name.startswith("gradients"):
                                        if not op.values()[0].name.startswith("softmax_cross_entropy_loss"):
                                            if not op.type == "Placeholder":
                                                tmp_list.append(op.values()[0].name)

        # Shorten the list
        if not tmp_list:
            return tmp_list

        result = [tmp_list[-1]]
        for name in reversed(tmp_list[:-1]):
            if result[0].split("/")[0] != name.split("/")[0]:
                result = [name] + result
        logger.info('Inferred %i hidden layers on TensorFlow classifier.', len(result))

        return result

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
        return self._layer_names

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
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            layer_tensor = graph.get_tensor_by_name(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])

        else:
            raise TypeError("Layer must be of type `str` or `int`. Received %s" % layer)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Create feed_dict
            feed_dict = {self._input_ph: x_preprocessed[begin:end]}
            feed_dict.update(self._feed_dict)

            # Run prediction for the current batch
            layer_output = self._sess.run(layer_tensor, feed_dict=feed_dict)
            results.append(layer_output)

        results = np.concatenate(results)

        return results

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self._feed_dict[self._learning] = train

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        # pylint: disable=E0611
        import os
        import shutil
        from tensorflow.python import saved_model
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

        if path is None:
            from art import DATA_PATH
            full_path = os.path.join(DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)

        if os.path.exists(full_path):
            shutil.rmtree(full_path)

        builder = saved_model.builder.SavedModelBuilder(full_path)
        signature = predict_signature_def(inputs={'SavedInputPhD': self._input_ph},
                                          outputs={'SavedOutputLogit': self._logits})
        builder.add_meta_graph_and_variables(sess=self._sess, tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()

        logger.info('Model saved in path: %s.', full_path)

    def __getstate__(self):
        """
        Use to ensure `TFClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time

        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state['_sess']
        del state['_logits']
        del state['_input_ph']
        state['_probs'] = self._probs.name

        if self._output_ph is not None:
            state['_output_ph'] = self._output_ph.name

        if self._loss is not None:
            state['_loss'] = self._loss.name

        if hasattr(self, '_loss_grads'):
            state['_loss_grads'] = self._loss_grads.name
        else:
            state['_loss_grads'] = False

        if self._learning is not None:
            state['_learning'] = self._learning.name

        if self._train is not None:
            state['_train'] = self._train.name

        if hasattr(self, '_logit_class_grads'):
            state['_logit_class_grads'] = [ts if ts is None else ts.name for ts in self._logit_class_grads]
        else:
            state['_logit_class_grads'] = False

        if hasattr(self, '_class_grads'):
            state['_class_grads'] = [ts if ts is None else ts.name for ts in self._class_grads]
        else:
            state['_class_grads'] = False

        model_name = str(time.time())
        state['model_name'] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state):
        """
        Use to ensure `TFClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Tensorflow
        # pylint: disable=E0611
        import os
        from art import DATA_PATH
        import tensorflow as tf
        from tensorflow.python.saved_model import tag_constants

        full_path = os.path.join(DATA_PATH, state['model_name'])

        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        loaded = tf.saved_model.loader.load(sess, [tag_constants.SERVING], full_path)

        # Recover session
        self._sess = sess

        # Recover logits
        logits_tensor_name = loaded.signature_def['predict'].outputs['SavedOutputLogit'].name
        self._logits = graph.get_tensor_by_name(logits_tensor_name)

        # Recover input_ph
        input_tensor_name = loaded.signature_def['predict'].inputs['SavedInputPhD'].name
        self._input_ph = graph.get_tensor_by_name(input_tensor_name)

        # Recover probability layer
        self._probs = graph.get_tensor_by_name(state['_probs'])

        # Recover output_ph if any
        if state['_output_ph'] is not None:
            self._output_ph = graph.get_tensor_by_name(state['_output_ph'])

        # Recover loss if any
        if state['_loss'] is not None:
            self._loss = graph.get_tensor_by_name(state['_loss'])

        # Recover loss_grads if any
        if state['_loss_grads']:
            self._loss_grads = graph.get_tensor_by_name(state['_loss_grads'])
        else:
            self.__dict__.pop('_loss_grads', None)

        # Recover learning if any
        if state['_learning'] is not None:
            self._learning = graph.get_tensor_by_name(state['_learning'])

        # Recover train if any
        if state['_train'] is not None:
            self._train = graph.get_operation_by_name(state['_train'])

        # Recover logit_class_grads if any
        if state['_logit_class_grads']:
            self._logit_class_grads = [ts if ts is None else graph.get_tensor_by_name(ts)
                                       for ts in state['_logit_class_grads']]
        else:
            self.__dict__.pop('_logit_class_grads', None)

        # Recover class_grads if any
        if state['_class_grads']:
            self._class_grads = [ts if ts is None else graph.get_tensor_by_name(ts) for ts in state['_class_grads']]
        else:
            self.__dict__.pop('_class_grads', None)

        self.__dict__.pop('model_name', None)

    def __repr__(self):
        repr_ = "%s(input_ph=%r, logits=%r, output_ph=%r, train=%r, loss=%r, learning=%r, " \
                "sess=%r, channel_index=%r, clip_values=%r, defences=%r, preprocessing=%r)" \
                % (self.__module__ + '.' + self.__class__.__name__,
                   self._input_ph, self._logits, self._output_ph, self._train, self._loss, self._learning, self._sess,
                   self.channel_index, self.clip_values, self.defences, self.preprocessing)

        return repr_
