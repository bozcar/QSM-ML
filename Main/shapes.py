from typing import Tuple

import tensorflow as tf

class shapes:
    """Images of 3D shapes stored as tensorflow tensors.
    
    Represents 3D objects as a grid of voxels stored in a tensorflor tensor. Voxels in the
    background have a value of zero, voxels in the object have non-zero values.

    Most constructors produce a single shape, with a constant value in the shape controlled
    by a weight parameter.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        The voxel values of the image and some associated information (shape, data type)

    Methods
    -------
    sphere(shape, centre, radius, weight=1.)
        Constructs the tensor representing a sphere with radius `radius` centred at point
        `centre` in a background with dimensions speicified by `shape`.
    cuboid(shape, small_corner, big_corner, weight=1.)
        Constructs the tensor representing the cuboid which has opposite corners at the 
        points specified by `big_corner` and `small_corner` in a background with dimensions
        speicified by `shape`.

    Notes
    -----
    The purpose of this class is to provide ways of generating 3D shapes for use as test 
    objects or training data.

    """
    def __init__(self, tensor: tf.Tensor):
        self.tensor = tensor

    def __str__(self):
        return str(self.tensor.numpy())

    @classmethod
    def sphere(cls, shape: Tuple[int], centre: Tuple[int], radius: float, weight: float = 1.):
        """Generates a 3D image of a shpere.

        An alternate constructor for the shape class.

        Creates a 3D image of a sphere with a radius of `radius` centred at `centre`. Voxels outside
        the sphere have a value of zero and voxels inside the sphere have a value of `weight`. `shape`
        determines the dimensions of the image.

        Parameters
        ----------
        shape : tuple of 3 ints
            Shape of the image, e.g., (3, 1, 2)
        centre : tuple of 3 ints
            Centre point of the sphere.
        radius : float
            Radius of the sphere.
        weight : float, optional
            The value of points inside the sphere. Defalut is 1.

        Returns
        -------
        out : shapes
            Image of a sphere with the given shape, centre and radius.

        See Also
        --------
        cuboid

        """
        x = tf.range(shape[0])
        y = tf.range(shape[1])
        z = tf.range(shape[2])

        X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')

        in_sphere = (X - centre[0])**2 + (Y - centre[1])**2 + (Z - centre[2])**2 < radius**2

        tensor = tf.where(in_sphere, tf.ones(shape), tf.zeros(shape)) * weight

        return cls(tensor)

    @classmethod
    def cuboid(cls, shape: Tuple[int], small_corner: Tuple[int], big_corner: Tuple[int], weight: float = 1):
        """Generates a 3D image of a cuboid.

        An alternate constructor for the shape class.

        Creates a 3D image of a cuboid which lies between the points `small_corner` and `big_corner`.

        Parameters
        ----------
        shape : tuple of 3 ints
            Shape of the image, e.g., (3, 1, 2)
        small_corner : tuple of 3 ints
            The corner of the cuboid closer to the origin.
        big_corner : tuple of 3 ints
            The corner of the cuboid further from the origin.
        weight : float, optional
            The value of points inside the sphere. Defalut is 1.

        Returns
        -------
        out : shapes
            Image of a cuboid with the given shape and bounding corners.

        See Also
        --------
        sphere
        
        """
        x = tf.range(shape[0])
        y = tf.range(shape[1])
        z = tf.range(shape[2])

        X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')

        tensor = tf.where(
            (small_corner[0] <= X) 
            & (X <= big_corner[0])
            & (small_corner[1] <= Y)
            & (Y <= big_corner[1])
            & (small_corner[2] <= Z)
            & (Z <= big_corner[2]), 
            tf.ones(shape), tf.zeros(shape)
        ) * weight
        
        return cls(tensor)