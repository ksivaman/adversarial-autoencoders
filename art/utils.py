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
Module providing convenience functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    # Conditional import of `torch` to avoid segmentation fault errors this framework generates at import
    import torch
except ImportError:
    logger.info('Could not import PyTorch in utilities.')


# -------------------------------------------------------------------------------------------- RANDOM NUMBER GENERATORS


def master_seed(seed):
    """
    Set the seed for all random number generators used in the library. This ensures experiments reproducibility and
    stable testing.

    :param seed: The value to be seeded in the random number generators.
    :type seed: `int`
    """
    import numbers
    import random

    if not isinstance(seed, numbers.Integral):
        raise TypeError('The seed for random number generators has to be an integer.')

    # Set Python seed
    random.seed(seed)

    # Set Numpy seed
    np.random.seed(seed)
    np.random.RandomState(seed)

    # Now try to set seed for all specific frameworks
    try:
        import tensorflow as tf

        logger.info('Setting random seed for TensorFlow.')
        tf.set_random_seed(seed)
    except ImportError:
        logger.info('Could not set random seed for TensorFlow.')

    try:
        import mxnet as mx

        logger.info('Setting random seed for MXNet.')
        mx.random.seed(seed)
    except ImportError:
        logger.info('Could not set random seed for MXNet.')

    try:
        logger.info('Setting random seed for PyTorch.')
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.info('Could not set random seed for PyTorch.')


# ----------------------------------------------------------------------------------------------------- MATHS UTILITIES


def projection(values, eps, norm_p):
    """
    Project `values` on the L_p norm ball of size `eps`.

    :param values: Array of perturbations to clip.
    :type values: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param norm_p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type norm_p: `int`
    :return: Values of `values` after projection.
    :rtype: `np.ndarray`
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        values_tmp = values_tmp * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1) + tol)),
                                                 axis=1)
    elif norm_p == 1:
        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1)
    elif norm_p == np.inf:
        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)
    else:
        raise NotImplementedError('Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.')

    values = values_tmp.reshape(values.shape)
    return values


def random_sphere(nb_points, nb_dims, radius, norm):
    """
    Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.

    :param nb_points: Number of random data points
    :type nb_points: `int`
    :param nb_dims: Dimensionality
    :type nb_dims: `int`
    :param radius: Radius
    :type radius: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    if norm == 1:
        a_tmp = np.zeros(shape=(nb_points, nb_dims + 1))
        a_tmp[:, -1] = np.sqrt(np.random.uniform(0, radius ** 2, nb_points))

        for i in range(nb_points):
            a_tmp[i, 1:-1] = np.sort(np.random.uniform(0, a_tmp[i, -1], nb_dims - 1))

        res = (a_tmp[:, 1:] - a_tmp[:, :-1]) * np.random.choice([-1, 1], (nb_points, nb_dims))
    elif norm == 2:
        # pylint: disable=E0611
        from scipy.special import gammainc

        a_tmp = np.random.randn(nb_points, nb_dims)
        s_2 = np.sum(a_tmp ** 2, axis=1)
        base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
        res = a_tmp * (np.tile(base, (nb_dims, 1))).T
    elif norm == np.inf:
        res = np.random.uniform(float(-radius), float(radius), (nb_points, nb_dims))
    else:
        raise NotImplementedError("Norm {} not supported".format(norm))

    return res


def original_to_tanh(x_original, clip_min, clip_max, tanh_smoother=0.999999):
    """
    Transform input from original to tanh space.

    :param x_original: An array with the input to be transformed.
    :type x_original: `np.ndarray`
    :param clip_min: Minimum clipping value.
    :type clip_min: `float` or `np.ndarray`
    :param clip_max: Maximum clipping value.
    :type clip_max: `float` or `np.ndarray`
    :param tanh_smoother: Scalar for multiplying arguments of arctanh to avoid division by zero.
    :type tanh_smoother: `float`
    :return: An array holding the transformed input.
    :rtype: `np.ndarray`
    """
    x_tanh = np.clip(x_original, clip_min, clip_max)
    x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
    x_tanh = np.arctanh(((x_tanh * 2) - 1) * tanh_smoother)
    return x_tanh


def tanh_to_original(x_tanh, clip_min, clip_max, tanh_smoother=0.999999):
    """
    Transform input from tanh to original space.

    :param x_tanh: An array with the input to be transformed.
    :type x_tanh: `np.ndarray`
    :param clip_min: Minimum clipping value.
    :type clip_min: `float` or `np.ndarray`
    :param clip_max: Maximum clipping value.
    :type clip_max: `float` or `np.ndarray`
    :param tanh_smoother: Scalar for dividing arguments of tanh to avoid division by zero.
    :type tanh_smoother: `float`
    :return: An array holding the transformed input.
    :rtype: `np.ndarray`
    """
    x_original = (np.tanh(x_tanh) / tanh_smoother + 1) / 2
    return x_original * (clip_max - clip_min) + clip_min


# --------------------------------------------------------------------------------------- LABELS MANIPULATION FUNCTIONS


def to_categorical(labels, nb_classes=None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes (possible labels)
    :type nb_classes: `int`
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels, dtype=np.int32)
    if not nb_classes:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def random_targets(labels, nb_classes):
    """
    Given a set of correct labels, randomly choose target labels different from the original ones. These can be
    one-hot encoded or integers.

    :param labels: The correct labels
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes for this model
    :type nb_classes: `int`
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    :rtype: `np.ndarray`
    """
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    result = np.zeros(labels.shape)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return to_categorical(result, nb_classes)


def least_likely_class(x, classifier):
    """
    Compute the least likely class predictions for sample `x`. This strategy for choosing attack targets was used in
    (Kurakin et al., 2016). See https://arxiv.org/abs/1607.02533.

    :param x: A data sample of shape accepted by `classifier`.
    :type x: `np.ndarray`
    :param classifier: The classifier used for computing predictions.
    :type classifier: `Classifier`
    :return: Least-likely class predicted by `classifier` for sample `x` in one-hot encoding.
    :rtype: `np.ndarray`
    """
    return to_categorical(np.argmin(classifier.predict(x), axis=1), nb_classes=classifier.nb_classes)


def second_most_likely_class(x, classifier):
    """
    Compute the second most likely class predictions for sample `x`. This strategy can be used for choosing target
    labels for an attack to improve its chances to succeed.

    :param x: A data sample of shape accepted by `classifier`.
    :type x: `np.ndarray`
    :param classifier: The classifier used for computing predictions.
    :type classifier: `Classifier`
    :return: Second most likely class predicted by `classifier` for sample `x` in one-hot encoding.
    :rtype: `np.ndarray`
    """
    return to_categorical(np.argpartition(classifier.predict(x), -2, axis=1)[:, -2], nb_classes=classifier.nb_classes)


def get_label_conf(y_vec):
    """
    Returns the confidence and the label of the most probable class given a vector of class confidences
    :param y_vec: (np.ndarray) vector of class confidences, nb of instances as first dimension
    :return: (np.ndarray, np.ndarray) confidences and labels
    """
    assert len(y_vec.shape) == 2

    confs, labels = np.amax(y_vec, axis=1), np.argmax(y_vec, axis=1)
    return confs, labels


def get_labels_np_array(preds):
    """
    Returns the label of the most probable class given a array of class confidences.

    :param preds: (np.ndarray) array of class confidences, nb of instances as first dimension
    :return: (np.ndarray) labels
    """
    preds_max = np.amax(preds, axis=1, keepdims=True)
    y = (preds == preds_max).astype(float)

    return y


def preprocess(x, y, nb_classes=10, clip_values=None):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :type x: `np.ndarray`
    :param y: Labels.
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset.
    :type nb_classes: `int`
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :type clip_values: `tuple(float, float)` or `tuple(np.ndarray, np.ndarray)`
    :return: Rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y


def compute_success(classifier, x_clean, labels, x_adv, targeted=False, batch_size=1):
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :type classifier: :class:`.Classifier`
    :param x_clean: Original clean samples.
    :type x_clean: `np.ndarray`
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :type labels: `np.ndarray`
    :param x_adv: Adversarial samples to be evaluated.
    :type x_adv: `np.ndarray`
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.s
    :type targeted: `bool`
    :param batch_size: Batch size
    :type batch_size: `int`
    :return: Percentage of successful adversarial samples.
    :rtype: `float`
    """
    adv_preds = np.argmax(classifier.predict(x_adv, batch_size=batch_size), axis=1)
    if targeted:
        rate = np.sum(adv_preds == np.argmax(labels, axis=1)) / x_adv.shape[0]
    else:
        preds = np.argmax(classifier.predict(x_clean, batch_size=batch_size), axis=1)
        rate = np.sum(adv_preds != preds) / x_adv.shape[0]

    return rate


# -------------------------------------------------------------------------------------------------------- IO FUNCTIONS


def load_cifar10(raw=False):
    """Loads CIFAR10 dataset from config.CIFAR10_PATH or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """

    def load_batch(fpath):
        """
        Utility function for loading CIFAR batches, as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        import sys
        from six.moves import cPickle

        with open(fpath, 'rb') as file_:
            if sys.version_info < (3,):
                content = cPickle.load(file_)
            else:
                content = cPickle.load(file_, encoding='bytes')
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode('utf8')] = value
                content = content_decoded
        data = content['data']
        labels = content['labels']

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    from art import DATA_PATH

    path = get_file('cifar-10-batches-py', extract=True, path=DATA_PATH,
                    url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Set channels last
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    min_, max_ = 0, 255
    if not raw:
        min_, max_ = 0., 1.
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_mnist(raw=False):
    """Loads MNIST dataset from `DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """
    from art import DATA_PATH

    path = get_file('mnist.npz', path=DATA_PATH, url='https://s3.amazonaws.com/img-datasets/mnist.npz')

    dict_mnist = np.load(path)
    x_train = dict_mnist['x_train']
    y_train = dict_mnist['y_train']
    x_test = dict_mnist['x_test']
    y_test = dict_mnist['y_test']
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0, 255
    if not raw:
        min_, max_ = 0., 1.
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_stl():
    """
    Loads the STL-10 dataset from `DATA_PATH` or downloads it if necessary.

    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """
    from os.path import join
    from art import DATA_PATH

    min_, max_ = 0., 1.

    # Download and extract data if needed
    path = get_file('stl10_binary', path=DATA_PATH, extract=True,
                    url='https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')

    with open(join(path, 'train_X.bin'), 'rb') as f_numpy:
        x_train = np.fromfile(f_numpy, dtype=np.uint8)
        x_train = np.reshape(x_train, (-1, 3, 96, 96))

    with open(join(path, 'test_X.bin'), 'rb') as f_numpy:
        x_test = np.fromfile(f_numpy, dtype=np.uint8)
        x_test = np.reshape(x_test, (-1, 3, 96, 96))

    # Set channel last
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    with open(join(path, 'train_y.bin'), 'rb') as f_numpy:
        y_train = np.fromfile(f_numpy, dtype=np.uint8)
        y_train -= 1

    with open(join(path, 'test_y.bin'), 'rb') as f_numpy:
        y_test = np.fromfile(f_numpy, dtype=np.uint8)
        y_test -= 1

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_iris(raw=False, test_set=.3):
    """
    Loads the UCI Iris dataset from `DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :param test_set: Proportion of the data to use as validation split. The value should be between o and 1.
    :type test_set: `float`
    :return: Entire dataset and labels.
    :rtype: `(np.ndarray, np.ndarray)`
    """
    from art import DATA_PATH, NUMPY_DTYPE

    # Download data if needed
    path = get_file('iris.data', path=DATA_PATH, extract=False,
                    url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

    data = np.loadtxt(path, delimiter=',', usecols=(0, 1, 2, 3), dtype=NUMPY_DTYPE)
    labels = np.loadtxt(path, delimiter=',', usecols=4, dtype=str)

    # Preprocess
    if not raw:
        label_map = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        labels = np.array([label_map[labels[i]] for i in range(labels.size)], dtype=np.int32)
        data, labels = preprocess(data, labels, nb_classes=3)
    min_, max_ = np.amin(data), np.amax(data)

    # Split training and test sets
    split_index = int((1 - test_set) * len(data) / 3)
    x_train = np.vstack((data[:split_index], data[50:50 + split_index], data[100:100 + split_index]))
    y_train = np.vstack((labels[:split_index], labels[50:50 + split_index], labels[100:100 + split_index]))

    if split_index >= 49:
        x_test, y_test = None, None
    else:

        x_test = np.vstack((data[split_index:50], data[50 + split_index:100], data[100 + split_index:]))
        y_test = np.vstack((labels[split_index:50], labels[50 + split_index:100], labels[100 + split_index:]))
        assert len(x_train) + len(x_test) == 150

        # Shuffle test set
        random_indices = np.random.permutation(len(y_test))
        x_test, y_test = x_test[random_indices], y_test[random_indices]

    # Shuffle training set
    random_indices = np.random.permutation(len(y_train))
    x_train, y_train = x_train[random_indices], y_train[random_indices]

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_dataset(name):
    """
    Loads or downloads the dataset corresponding to `name`. Options are: `mnist`, `cifar10` and `stl10`.

    :param name: Name of the dataset
    :type name: `str`
    :return: The dataset separated in training and test sets as `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    :raises NotImplementedError: If the dataset is unknown.
    """
    if "mnist" in name:
        return load_mnist()
    if "cifar10" in name:
        return load_cifar10()
    if "stl10" in name:
        return load_stl()
    if "iris" in name:
        return load_iris()

    raise NotImplementedError("There is no loader for dataset '{}'.".format(name))


def _extract(full_path, path):
    import tarfile
    import zipfile
    import shutil

    if full_path.endswith('tar'):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith('tar.gz'):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith('zip'):
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(filename, url, path=None, extract=False):
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :type filename: `str`
    :param url: Download URL.
    :type url: `str`
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :type: `str`
    :param extract: If true, tries to extract the archive.
    :type extract: `bool`
    :return: Path to the downloaded file.
    :rtype: `str`
    """
    if path is None:
        from art import DATA_PATH
        path_ = os.path.expanduser(DATA_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join('/tmp', '.art')
    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + '.tar.gz'
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info('Downloading data from %s', url)
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                from six.moves.urllib.error import HTTPError, URLError
                from six.moves.urllib.request import urlretrieve

                urlretrieve(url, full_path)
            except HTTPError as exception:
                raise Exception(error_msg.format(url, exception.code, exception.msg))
            except URLError as exception:
                raise Exception(error_msg.format(url, exception.errno, exception.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path


def make_directory(dir_path):
    """
    Creates the specified tree of directories if needed.

    :param dir_path: (str) directory or file path
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clip_and_round(x, clip_values, round_samples):
    """
    Rounds the input to the correct level of granularity.
    Useful to ensure data passed to classifier can be represented
    in the correct domain, e.g., [0, 255] integers verses [0,1]
    or [0, 255] floating points.

    :param x: Sample input with shape as expected by the model.
    :type x: `np.ndarray`
    :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
           for features, or `None` if no clipping should be performed.
    :type clip_values: `tuple`
    :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to
           disable.
    :type round_samples: `float`
    """
    if round_samples == 0:
        return x
    if clip_values is not None:
        np.clip(x, clip_values[0], clip_values[1], out=x)
    x = np.around(x / round_samples) * round_samples
    return x
