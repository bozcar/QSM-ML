from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from keras import layers

class ConvDipole(layers.Layer):
    r"""Convolve with the dipole kernel.

    Convolve an image with a dipole kernel with the same dimensions.

    Parameters
    ----------
    name : str, optional
        An optional name for the layer

    Notes
    -----
    The dipole kernel D is defined in k-space as:

    .. math::
    
        D = 
        \begin{cases}
        \frac{1}{3}-\frac{k_z^2}{k^2},\\
        0,
        \end{cases}
        \text{\quad\parbox{0.4\linewidth}{ for $k\neq0$\\ for $k=0$}}

    where :math:`k_x, k_y\text{ and }k_z` are the coordinates in k-space and 
    :math:`k^2 = k_x^2 + k_y^2 + k_z^2`. [1]

    References
    ----------
    .. [1] E Mark Haake; Saifeng Liu; Sagar Buch; Weili Zheng; Dongmei Wu; Yongquan Ye, 
       "Quantitative susceptibility mapping: current status and future directions" 
       Magnetic Resonance Imaging 33, 2015

    """
    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.is_built = False
    
    @tf.function
    def call(self, img: tf.Tensor) -> tf.Tensor:
        """Convolve an input image with the dipole kernel.

        Multiplies `img` by `kernel` in k-space by fourier transforming the 
        image before multiplication. This is done as the kernel is defined in 
        k-space, not image space.

        Parameters
        ----------
        img : tensorflow.Tensor
            Image to be convolved with the dipole kernel
        
        Returns
        -------
        conv : tensorflow.Tensor
            The result of convolving the image with the dipole kernel
        
        """
        # Wait until an image is passed to the layer to generate 
        # a dipole kernel with appropriate dimensions
        if not self.is_built:
            n_imgs, *shape = img.shape
            if not len(shape) == 3:
                raise ValueError(f"Input image has {len(shape)} dimensions, expected 3")
            self.kernel = self.generate_dipole(shape)
            self.kernel = tf.expand_dims(self.kernel, 0) # Add n_imgs dimension
            self.kernel = tf.tile(self.kernel, [n_imgs, 1, 1, 1])
            self.is_built = True

        kimg = tf.signal.fft3d(img)
        conv = tf.signal.ifft3d(kimg * self.kernel)
        return conv

    @staticmethod
    def generate_dipole(shape: Tuple[int]):
        """Generate a dipole kernel in k-space with the given `shape`."""
        ONE_THIRD = 1/3

        x = tf.linspace(-1, 1, shape[0])
        y = tf.linspace(-1, 1, shape[1])
        z = tf.linspace(-1, 1, shape[2])

        vx, vy, vz = tf.meshgrid(x, y, z, indexing='ij')

        denom = vx**2 + vy**2 + vz**2
        z_squared = vz**2

        kernel = tf.where(denom!=0, ONE_THIRD-(z_squared/denom), tf.zeros(shape, dtype = tf.float64))
        kernel = tf.signal.fftshift(kernel)
        kernel = tf.cast(kernel, tf.complex64) # Required for complatibility with fourier transformation

        return kernel

class WeightedSubtract(layers.Layer):
    """Subtract some change from an image with a trainable scalar weight `tau`.

    A layer for chaning an image by subtracting a set of intensities from the image.
    It has a trainable scalar weight `tau` with a default value of 2.

    Parameters
    ----------
    tau : float, optional
        A scalar weight for the subtraction. Default 2
    name : str, optionsal
        An optional name for the layer

    Notes
    -----
    This layer is used to subtract a model-defined gradient from an image and train the 
    step size.
    
    """
    def __init__(self, tau: float = 2, name: str = None) -> None:
        super().__init__(name=name)
        self.tau = tf.Variable(
            initial_value=tau,
            dtype=tf.complex64,
            trainable=True
        )
    
    @tf.function
    def call(self, img, diff):
        return img - (self.tau * diff)