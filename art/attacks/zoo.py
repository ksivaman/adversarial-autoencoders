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
This module implements the zeroth-order optimization attack `ZooAttack`. This is a black-box attack. This attack is a
variant of the Carlini and Wagner attack which uses ADAM coordinate descent to perform numerical estimation of
gradients.

Paper link:
    https://arxiv.org/abs/1708.03999.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.ndimage import zoom

from art import NUMPY_DTYPE
from art.attacks.attack import Attack
from art.utils import compute_success, get_labels_np_array

logger = logging.getLogger(__name__)


class ZooAttack(Attack):
    """
    The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). This attack is a variant of the
    C&W attack which uses ADAM coordinate descent to perform numerical estimation of gradients.
    Paper link: https://arxiv.org/abs/1708.03999.
    """
    attack_params = Attack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter',
                                            'binary_search_steps', 'initial_const', 'abort_early', 'use_resize',
                                            'use_importance', 'nb_parallel', 'batch_size']

    def __init__(self, classifier, confidence=0.0, targeted=False, learning_rate=1e-2, max_iter=10,
                 binary_search_steps=1, initial_const=1e-3, abort_early=True, use_resize=True, use_importance=True,
                 nb_parallel=128, batch_size=1):
        """
        Create a ZOO attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :type learning_rate: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :type binary_search_steps: `int`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :type initial_const: `float`
        :param abort_early: `True` if gradient descent should be abandoned when it gets stuck.
        :type abort_early: `bool`
        :param use_resize: `True` if to use the resizing strategy from the paper: first, compute attack on inputs
               resized to 32x32, then increase size if needed to 64x64, followed by 128x128.
        :type use_resize: `bool`
        :param use_importance: `True` if to use importance sampling when choosing coordinates to update.
        :type use_importance: `bool`
        :param nb_parallel: Number of coordinate updates to run in parallel. A higher value for `nb_parallel` should
               be preferred over a large batch size.
        :type nb_parallel: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated. Small batch sizes are
               encouraged for ZOO, as the algorithm already runs `nb_parallel` coordinate updates in parallel for each
               sample. The batch size is a multiplier of `nb_parallel` in terms of memory consumption.
        :type batch_size: `int`
        """
        super(ZooAttack, self).__init__(classifier)

        if len(classifier.input_shape) == 1:
            raise ValueError('Feature vectors detected. The ZOO attack can only be applied to data with spatial'
                             'dimensions.')

        kwargs = {
            'confidence': confidence,
            'targeted': targeted,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'binary_search_steps': binary_search_steps,
            'initial_const': initial_const,
            'abort_early': abort_early,
            'use_resize': use_resize,
            'use_importance': use_importance,
            'nb_parallel': nb_parallel,
            'batch_size': batch_size
        }
        self.set_params(**kwargs)

        # Initialize some internal variables
        self._init_size = 32
        if self.abort_early:
            self._early_stop_iters = self.max_iter // 10 if self.max_iter >= 10 else self.max_iter
        self.nb_parallel = nb_parallel

        # Initialize noise variable to zero
        if self.use_resize:
            if self.classifier.channel_index == 3:
                dims = (batch_size, self._init_size, self._init_size, self.classifier.input_shape[-1])
            elif self.classifier.channel_index == 1:
                dims = (batch_size, self.classifier.input_shape[0], self._init_size, self._init_size)
            self._current_noise = np.zeros(dims, dtype=NUMPY_DTYPE)
        else:
            self._current_noise = np.zeros((batch_size,) + self.classifier.input_shape, dtype=NUMPY_DTYPE)
        self._sample_prob = np.ones(self._current_noise.size, dtype=NUMPY_DTYPE) / self._current_noise.size

        self.adam_mean = None
        self.adam_var = None
        self.adam_epochs = None

    def _loss(self, x, x_adv, target, c_weight):
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param c_weight: Weight of the loss term aiming for classification as target.
        :type c_weight: `float`
        :return: A tuple holding the current logits, `L_2` distortion and overall loss.
        :rtype: `(float, float, float)`
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x_adv.shape[0], -1), axis=1)
        ratios = [1] + [int(new_size) / int(old_size)
                        for new_size, old_size in zip(self.classifier.input_shape, x.shape[1:])]
        preds = self.classifier.predict(np.array(zoom(x_adv, zoom=ratios)), batch_size=self.batch_size)
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        if self.targeted:
            # If targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, 0)
        else:
            # If untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, 0)

        return preds, l2dist, c_weight * loss + l2dist

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target labels. Otherwise, the targets are the
                  original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        # ZOO can probably be extended to feature vectors if no zooming or resizing is applied
        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. The ZOO attack can only be applied to data with spatial'
                             'dimensions.')

        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, logits=False, batch_size=self.batch_size))

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv = []
        for batch_id in range(nb_batches):
            logger.debug('Processing batch %i out of %i', batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv.append(res)
        x_adv = np.vstack(x_adv)

        # Apply clip
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)

        # Log success rate of the ZOO attack
        logger.info('Success rate of ZOO attack: %.2f%%',
                    100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size))

        return x_adv

    def _generate_batch(self, x_batch, y_batch):
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :type x_batch: `np.ndarray`
        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :return: A batch of adversarial examples.
        :rtype: `np.ndarray`
        """
        # Initialize binary search
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 1e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug('Binary search step %i out of %i (c_mean==%f)', bss, self.binary_search_steps,
                         np.mean(c_current))

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self._update_const(y_batch, best_label, c_current, c_lower_bound,
                                                                         c_upper_bound)

        return o_best_attack

    def _update_const(self, y_batch, best_label, c_batch, c_lower_bound, c_upper_bound):
        """
        Update constant `c_batch` from the ZOO objective. This characterizes the trade-off between attack strength and
        amount of noise introduced.

        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :param best_label: A batch of best labels.
        :type best_label: `np.ndarray`
        :param c_batch: A batch of constants.
        :type c_batch: `np.ndarray`
        :param c_lower_bound: A batch of lower bound constants.
        :type c_lower_bound: `np.ndarray`
        :param c_upper_bound: A batch of upper bound constants.
        :type c_upper_bound: `np.ndarray`
        :return: A tuple of three batches of updated constants and lower/upper bounds.
        :rtype: `tuple`
        """

        def compare(object1, object2):
            return object1 == object2 if self.targeted else object1 != object2

        comparison = [compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf for i in
                      range(len(c_batch))]
        for i, comp in enumerate(comparison):
            if comp:
                # Successful attack
                c_upper_bound[i] = min(c_upper_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
            else:
                # Failure attack
                c_lower_bound[i] = max(c_lower_bound[i], c_batch[i])
                c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2 if c_upper_bound[i] < 1e9 else c_batch[i] * 10

        return c_batch, c_lower_bound, c_upper_bound

    def _generate_bss(self, x_batch, y_batch, c_batch):
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :type x_batch: `np.ndarray`
        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :param c_batch: A batch of constants.
        :type c_batch: `np.ndarray`
        :return: A tuple of best elastic distances, best labels, best attacks
        :rtype: `tuple`
        """

        def compare(object1, object2):
            return object1 == object2 if self.targeted else object1 != object2

        x_orig = x_batch.astype(NUMPY_DTYPE)
        fine_tuning = np.full(x_batch.shape[0], False, dtype=bool)
        prev_loss = 1e6 * np.ones(x_batch.shape[0])
        prev_l2dist = np.zeros(x_batch.shape[0])

        # Resize and initialize Adam
        if self.use_resize:
            x_orig = self._resize_image(x_orig, self._init_size, self._init_size, True)
            assert (x_orig != 0).any()
            x_adv = x_orig.copy()
        else:
            x_orig = x_batch
            self._reset_adam(np.prod(self.classifier.input_shape))
            self._current_noise.fill(0)

        # Initialize best distortions, best changed labels and best attacks
        best_dist = np.inf * np.ones(x_adv.shape[0])
        best_label = -np.inf * np.ones(x_adv.shape[0])
        best_attack = [x_adv[i] for i in range(x_adv.shape[0])]

        for iter_ in range(self.max_iter):
            logger.debug('Iteration step %i out of %i', iter_, self.max_iter)

            # Upscaling for very large number of iterations
            if self.use_resize:
                if iter_ == 2000:
                    x_adv = self._resize_image(x_adv, 64, 64)
                    x_orig = zoom(x_orig, [1, x_adv.shape[1] / x_orig.shape[1],
                                           x_adv.shape[2] / x_orig.shape[2], x_adv.shape[3] / x_orig.shape[3]])
                elif iter_ == 10000:
                    x_adv = self._resize_image(x_adv, 128, 128)
                    x_orig = zoom(x_orig, [1, x_adv.shape[1] / x_orig.shape[1],
                                           x_adv.shape[2] / x_orig.shape[2], x_adv.shape[3] / x_orig.shape[3]])

            # Compute adversarial examples and loss
            x_adv = self._optimizer(x_adv, y_batch, c_batch)
            preds, l2dist, loss = self._loss(x_orig, x_adv, y_batch, c_batch)

            # Reset Adam if a valid example has been found to avoid overshoot
            mask_fine_tune = (~fine_tuning) & (loss == l2dist) & (prev_loss != prev_l2dist)
            fine_tuning[mask_fine_tune] = True
            self._reset_adam(self.adam_mean.size, np.repeat(mask_fine_tune, x_adv[0].size))
            prev_l2dist = l2dist

            # Abort early if no improvement is obtained
            if self.abort_early and iter_ % self._early_stop_iters == 0:
                if (loss > .9999 * prev_loss).all():
                    break
                prev_loss = loss

            # Adjust the best result
            labels_batch = np.argmax(y_batch, axis=1)
            for i, (dist, pred) in enumerate(zip(l2dist, np.argmax(preds, axis=1))):
                if dist < best_dist[i] and compare(pred, labels_batch[i]):
                    best_dist[i] = dist
                    best_attack[i] = x_adv[i]
                    best_label[i] = pred

        # Resize images to original size before returning
        best_attack = np.array(best_attack)
        if self.use_resize:
            if self.classifier.channel_index == 3:
                best_attack = zoom(best_attack, [1, int(x_batch.shape[1]) / best_attack.shape[1],
                                                 int(x_batch.shape[2]) / best_attack.shape[2], 1])
            elif self.classifier.channel_index == 1:
                best_attack = zoom(best_attack, [1, 1, int(x_batch.shape[2]) / best_attack.shape[2],
                                                 int(x_batch.shape[2]) / best_attack.shape[3]])

        return best_dist, best_label, best_attack

    def _optimizer(self, x, targets, c_batch):
        # Variation of input for computing loss, same as in original implementation
        tol = 1e-4
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        # Sample indices to prioritize for optimization
        if self.use_importance and np.unique(self._sample_prob).size != 1:
            indices = np.random.choice(coord_batch.shape[-1] * x.shape[0],
                                       self.nb_parallel * self._current_noise.shape[0],
                                       replace=False, p=self._sample_prob.flatten()) % coord_batch.shape[-1]
        else:
            indices = np.random.choice(coord_batch.shape[-1] * x.shape[0],
                                       self.nb_parallel * self._current_noise.shape[0],
                                       replace=False) % coord_batch.shape[-1]

        # Create the batch of modifications to run
        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += tol
            coord_batch[2 * i + 1, indices[i]] -= tol

        # Compute loss for all samples and coordinates, then optimize
        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        expanded_targets = np.repeat(targets, 2 * self.nb_parallel, axis=0).reshape((-1,) + targets.shape[1:])
        expanded_c = np.repeat(c_batch, 2 * self.nb_parallel)
        _, _, loss = self._loss(expanded_x, expanded_x + coord_batch.reshape(expanded_x.shape), expanded_targets,
                                expanded_c)
        self._current_noise = self._optimizer_adam_coordinate(loss, indices, self.adam_mean, self.adam_var,
                                                              self._current_noise, self.learning_rate, self.adam_epochs,
                                                              True)

        if self._current_noise.shape[2] > self._init_size:
            self._sample_prob = self._get_prob(self._current_noise).flatten()

        return x + self._current_noise

    def _optimizer_adam_coordinate(self, losses, index, mean, var, current_noise, learning_rate, adam_epochs, proj):
        """
        Implementation of the ADAM optimizer for coordinate descent.
        """
        beta1, beta2 = .9, .999

        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / .0002 for i in range(0, len(losses), 2)])

        # ADAM update
        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads ** 2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            current_noise[index] = np.clip(current_noise[index], clip_min, clip_max)

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars, indices=None):
        # If variables are already there and at the right size, reset values
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)
                self.adam_epochs.fill(1)
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0
                self.adam_epochs[indices] = 1
        else:
            # Allocate Adam variables
            self.adam_mean = np.zeros(nb_vars, dtype=NUMPY_DTYPE)
            self.adam_var = np.zeros(nb_vars, dtype=NUMPY_DTYPE)
            self.adam_epochs = np.ones(nb_vars, dtype=np.int32)

    def _resize_image(self, x, size_x, size_y, reset=False):
        if self.classifier.channel_index == 3:
            dims = (x.shape[0], size_x, size_y, x.shape[-1])
        elif self.classifier.channel_index == 1:
            dims = (x.shape[0], x.shape[1], size_x, size_y)
        nb_vars = np.prod(dims)

        if reset:
            # Reset variables to original size and value
            if dims == x.shape:
                resized_x = x
                self._current_noise.fill(0)
            else:
                resized_x = zoom(x, (1, dims[1] / x.shape[1], dims[2] / x.shape[2], dims[3] / x.shape[3]))
                self._current_noise = np.zeros(dims, dtype=NUMPY_DTYPE)
            self._sample_prob = np.ones(nb_vars, dtype=NUMPY_DTYPE) / nb_vars
        else:
            # Rescale variables and reset values
            resized_x = zoom(x, (1, dims[1] / x.shape[1], dims[2] / x.shape[2], dims[3] / x.shape[3]))
            self._sample_prob = self._get_prob(self._current_noise, double=True).flatten()
            self._current_noise = np.zeros(dims, dtype=NUMPY_DTYPE)

        # Reset Adam
        self._reset_adam(nb_vars)

        return resized_x

    def _get_prob(self, prev_noise, double=False):
        dims = list(prev_noise.shape)

        # Double size if needed
        if double:
            dims = [2 * size if i not in [0, self.classifier.channel_index] else size for i, size in enumerate(dims)]

        prob = np.empty(shape=dims, dtype=np.float32)
        image = np.abs(prev_noise)

        for channel in range(prev_noise.shape[self.classifier.channel_index]):
            if self.classifier.channel_index == 3:
                image_pool = self._max_pooling(image[:, :, :, channel], dims[1] // 8)
                if double:
                    prob[:, :, :, channel] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, :, :, channel] = image_pool
            elif self.classifier.channel_index == 1:
                image_pool = self._max_pooling(image[:, channel, :, :], dims[2] // 8)
                if double:
                    prob[:, channel, :, :] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, channel, :, :] = image_pool

        prob /= np.sum(prob)

        return prob

    @staticmethod
    def _max_pooling(image, kernel_size):
        img_pool = np.copy(image)
        for i in range(0, image.shape[1], kernel_size):
            for j in range(0, image.shape[2], kernel_size):
                img_pool[:, i:i + kernel_size, j:j + kernel_size] = np.max(
                    image[:, i:i + kernel_size, j:j + kernel_size],
                    axis=(1, 2), keepdims=True)

        return img_pool

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :type learning_rate: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :type binary_search_steps: `int`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :type initial_const: `float`
        :param abort_early: `True` if gradient descent should be abandoned when it gets stuck.
        :type abort_early: `bool`
        :param use_resize: `True` if to use the resizing strategy from the paper: first, compute attack on inputs
               resized to 32x32, then increase size if needed to 64x64, followed by 128x128.
        :type use_resize: `bool`
        :param use_importance: `True` if to use importance sampling when choosing coordinates to update.
        :type use_importance: `bool`
        :param nb_parallel: Number of coordinate updates to run in parallel. A higher value for `nb_parallel` should
               be preferred over a large batch size.
        :type nb_parallel: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated. Small batch sizes are
               encouraged for ZOO, as the algorithm already runs `nb_parallel` coordinate updates in parallel for each
               sample. The batch size is a multiplier of `nb_parallel` in terms of memory consumption.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(ZooAttack, self).set_params(**kwargs)

        if not isinstance(self.binary_search_steps, (int, np.int)) or self.binary_search_steps < 0:
            raise ValueError('The number of binary search steps must be a non-negative integer.')

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError('The number of iterations must be a non-negative integer.')

        if not isinstance(self.nb_parallel, (int, np.int)) or self.nb_parallel < 1:
            raise ValueError('The number of parallel coordinates must be an integer greater than zero.')

        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size < 1:
            raise ValueError('The batch size must be an integer greater than zero.')

        return True
