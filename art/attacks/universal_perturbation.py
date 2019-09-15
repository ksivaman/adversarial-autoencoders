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
This module implements the universal adversarial perturbations attack `UniversalPerturbation`. This is a white-box
attack.

Paper link:
    https://arxiv.org/abs/1610.08401
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random

import numpy as np

from art.attacks.attack import Attack
from art.utils import projection

logger = logging.getLogger(__name__)


class UniversalPerturbation(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a fixed perturbation to be applied to all
    future inputs. To this end, it can use any adversarial attack method. Paper link: https://arxiv.org/abs/1610.08401
    """
    attacks_dict = {'carlini': 'art.attacks.carlini.CarliniL2Method',
                    'carlini_inf': 'art.attacks.carlini.CarliniLInfMethod',
                    'deepfool': 'art.attacks.deepfool.DeepFool',
                    'ead': 'art.attacks.elastic_net.ElasticNet',
                    'fgsm': 'art.attacks.fast_gradient.FastGradientMethod',
                    'bim': 'art.attacks.iterative_method.BasicIterativeMethod',
                    'pgd': 'art.attacks.projected_gradient_descent.ProjectedGradientDescent',
                    'newtonfool': 'art.attacks.newtonfool.NewtonFool',
                    'jsma': 'art.attacks.saliency_map.SaliencyMapMethod',
                    'vat': 'art.attacks.virtual_adversarial.VirtualAdversarialMethod'
                    }
    attack_params = Attack.attack_params + ['attacker', 'attacker_params', 'delta', 'max_iter', 'eps', 'norm']

    def __init__(self, classifier, attacker='deepfool', attacker_params=None, delta=0.2, max_iter=20, eps=10.0,
                 norm=np.inf):
        """
        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param attacker: Adversarial attack name. Default is 'deepfool'. Supported names: 'carlini', 'carlini_inf',
                         'deepfool', 'fgsm', 'bim', 'pgd', 'margin', 'ead', 'newtonfool', 'jsma', 'vat'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: desired accuracy
        :type delta: `float`
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param norm: Order of the norm. Possible values: np.inf, 2 (default is np.inf)
        :type norm: `int`
        """
        super(UniversalPerturbation, self).__init__(classifier)
        kwargs = {'attacker': attacker,
                  'attacker_params': attacker_params,
                  'delta': delta,
                  'max_iter': max_iter,
                  'eps': eps,
                  'norm': norm
                  }
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: An array with the original labels to be predicted.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        logger.info('Computing universal perturbation based on %s attack.', self.attacker)

        # Init universal perturbation
        noise = 0
        fooling_rate = 0.0
        nb_instances = len(x)

        # Instantiate the middle attacker and get the predicted labels
        attacker = self._get_attack(self.attacker, self.attacker_params)
        pred_y = self.classifier.predict(x, logits=False, batch_size=1)
        pred_y_max = np.argmax(pred_y, axis=1)

        # Start to generate the adversarial examples
        nb_iter = 0
        while fooling_rate < 1. - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, ex in enumerate(x[rnd_idx]):
                x_i = ex[None, ...]

                current_label = np.argmax(self.classifier.predict(x_i + noise, logits=True)[0])
                original_label = np.argmax(pred_y[rnd_idx][j])

                if current_label == original_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(x_i + noise)
                    new_label = np.argmax(self.classifier.predict(adv_xi, logits=True)[0])

                    # If the class has changed, update v
                    if current_label != new_label:
                        noise = adv_xi - x_i

                        # Project on L_p ball
                        noise = projection(noise, self.eps, self.norm)
            nb_iter += 1

            # Apply attack and clip
            x_adv = x + noise
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)

            # Compute the error rate
            y_adv = np.argmax(self.classifier.predict(x_adv, logits=False, batch_size=1), axis=1)
            fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances

        self.fooling_rate = fooling_rate
        self.converged = nb_iter < self.max_iter
        self.noise = noise
        logger.info('Success rate of universal perturbation attack: %.2f%%', fooling_rate)

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param attacker: Adversarial attack name. Default is 'deepfool'. Supported names: 'carlini', 'deepfool', 'fgsm',
                'newtonfool', 'jsma', 'vat'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: desired accuracy
        :type delta: `float`
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param norm: Order of the norm. Possible values: np.inf, 2 (default is np.inf)
        :type norm: `int`
        """
        super(UniversalPerturbation, self).set_params(**kwargs)

        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")

        return True

    def _get_attack(self, a_name, params=None):
        """
        Get an attack object from its name.

        :param a_name: attack name.
        :type a_name: `str`
        :param params: attack params.
        :type params: `dict`
        :return: attack object
        :rtype: `object`
        """
        try:
            attack_class = self._get_class(self.attacks_dict[a_name])
            a_instance = attack_class(self.classifier)

            if params:
                a_instance.set_params(**params)

            return a_instance

        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name))

    @staticmethod
    def _get_class(class_name):
        """
        Get a class module from its name.

        :param class_name: Full name of a class.
        :type class_name: `str`
        :return: The class `module`.
        :rtype: `module`
        """
        sub_mods = class_name.split(".")
        module_ = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        class_module = getattr(module_, sub_mods[-1])

        return class_module
