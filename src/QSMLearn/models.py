from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from .layers import *


def NDIGrad(
    sus: tf.Tensor, 
    phase: tf.Tensor, 
    weight: tf. Tensor, 
    conv: ConvDipole
) -> tf.Tensor:
    """The gradient of the NDI cost function."""
    w_squared = weight * weight
    diff = conv(sus) - phase
    return conv(w_squared * tf.math.sin(diff))


class NDIModel:
    r"""Base class for implementing Nonlinear Dipole Inversion (NDI).

    This class allows for easy implementation of the NDI algorithm as a keras model.
    This allows variables like the step size `tau` of the model to be trained to 
    find optimal parameters for the model. 
    
    To use this class, define a subclass which where it's call function passes a function 
    defining how perform gradient decent steps into the parent calsses call function as 
    the `do_steps` parameter.

    Parameters
    ----------
    name : str, optional
        An optional name for the model

    Notes
    -----
    This class is intended to implemet unregularised NDI. This algorithm seeks to minimize 
    the cost function 
    
    .. math::
    
        f\left(\vec{\chi}\right)=\left|\left|W\left(e^{iD\vec{\chi}}-e^{i\vec{\phi}}\right)\right|\right|_2^2

    where :math:`\chi` is the distribution of susceptibility in the image, :math:`W` is a diagonal 
    weighting matrix (usually a mask covering a region of interest), :math:`D\chi` is the convolution 
    of the dipole kernel with the susceptibility distribution and :math:`\phi` is the input phase image.

    The gradient of this cost function :math:`\nabla_{\vec{\chi}}f\left(\vec{\chi}\right)` 
    has been derived analytically as:

    .. math::

        \nabla_{\vec{\chi}}f\left(\vec{\chi}\right) = 2D^{T}W^{T}W\sin\left(D\vec{\chi}-\vec{\phi}\right)

    [1] It is this gradient which is calculated by the method `__NDIGrad`.

    References
    ----------
    .. [1] Polak, Daniel; Chatnuntawech, Itthi; Yoon, Jaeyeon; Iyer, Siddharth Srinivasan; Milovic, 
       Carlos; Lee, Jongho; Bachert, Peter; Adalsteinsson, Elfar; Setsompop, Kawin; Bilgic, Berkin, 
       "Nonlinear dipole inversion (NDI) enables robust quantitative susceptibility mapping (QSM)"
       NMR in Biomedicine, vol. 12, pp. e4271, 2020.

    """
    def __init__(self):
        self.dipole_convolution = ConvDipole()

    def __NDIGrad(self, sus, phase, weight) -> tf.Tensor:
        """The gradient of the NDI cost function."""
        w_squared = weight * weight
        diff = self.dipole_convolution(sus) - phase
        return self.dipole_convolution(w_squared * tf.math.sin(diff))


class FixedStepNDI(keras.Model):
    """NDI model with one trainbable step size.

    Class for training the step size of the NDI model. It contains one `WeightedSubtract` layer
    which is applied a nubmer of times controlled by the parameter `iters`.

    Parameters
    ----------
    iters : int
        The number of iterations of the NDI model which will be applied.
    name : str, optionsal
        An optional name for the model
    
    """
    def __init__(self, iters: int, name: str = None, init_step: float = 2) -> None:
        super().__init__(name)
        self.iters = iters
        self.dipole_convolution = ConvDipole()
        self.step = WeightedSubtract(tau=init_step)

    @tf.function
    def call(self, t: tf.Tensor) -> tf.Tensor:
        """Applies the `WeightedSubtract` layer `iters` times."""
        phase = t[:, 0, :, :, :]
        weight = t[:, 1, :, :, :]
        # Cast to complex numbers to allow for fourier transforms
        sus = tf.zeros(tf.shape(phase))

        for _ in range(self.iters):
            sus = self.step(sus, NDIGrad(sus, phase, weight, self.dipole_convolution))
        
        return sus


class VariableStepNDI(keras.Model):
    """NDI model where each step size can be trained independently.

    Class for training an NDI model where the step size can vary on each 
    iteration. It contains a nubmer of `WeightedSubtract` layers controlled 
    by the parameter `iters`

    Parameters
    ----------
    iters : int
        The number of iterations of the NDI model which will be applied.
    name : str, optionsal
        An optional name for the model
    
    """
    def __init__(self, iters: int, name: str = None) -> None:
        super().__init__(name)
        self.dipole_convolution = ConvDipole()
        self.steps = [WeightedSubtract(name=f"setp{i}") for i in range(iters)]

    @tf.function
    def call(self, phase: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        """Applies each of the `iters` `WeightedSubtract` layers."""
        sus = tf.cast(tf.zeros(tf.shape(phase)), dtype=tf.complex64)
        pha = tf.cast(phase, dtype=tf.complex64)
        wei = tf.cast(weight, dtype=tf.complex64)

        i = 0
        for step in self.steps:
            sus = step(sus, NDIGrad(sus, pha, wei, self.dipole_convolution))
            if i % 10 == 0:
                print(f"Iteration {i+1}/{len(self.steps)} complete.")
            i += 1
        
        return sus