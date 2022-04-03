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
    def __init__(self, iters: int, *, name: str = None, initial_step: float = 2) -> None:
        super().__init__(name)
        self.iters = iters
        self.dipole_convolution = ConvDipole()
        self.step = WeightedSubtract(tau=initial_step)

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