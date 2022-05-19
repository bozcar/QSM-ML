from math import floor
from pathlib import Path

import numpy as np

from .utils import deformation_field_from_affine_matrix, resample_with_deformation_field


class Shapes:
    def __init__(self, dist: np.ndarray) -> None:
        """A simple geometric distribution of magnetic susceptibility.
        
        """
        self._dist = dist
        self._shape = dist.shape

    def __add__(self, other):
        dist_sum = self.dist + other.dist
        return Shapes(dist_sum)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # Properties
    @property
    def dist(self):
        return self._dist

    @property
    def shape(self):
        return self._shape

    # Constructors
    @classmethod
    def sphere(
        cls,
        shape: tuple[int],
        value: float = 1
    ):
        """Constructor to generate a sphere.

        Generates a sphere of susceptibility `value` in the centre of the 
        area specified by `shape`. The diameter of the sphere is half the 
        length of the shortest side.

        Parameters
        ----------
        shape : tuple[int, int, int]
            The resolution of the image in (x, y, z)
        
        value : float
            The value of the susceptibility inside the sphere

        Returns
        -------
        Shapes
            A `Shapes` object containing a sphere.

        See Also
        --------
        Shapes.cylinder
            Constructor to generate a cylinder.
        
        Shapes.cube
            Constructor to generate a cube.

        Notes
        -----
        To change the size/position of the sphere, or to produce 
        an ellipsoid, apply the appropriate affine transformation.
        
        """
        shortest = min(shape)
        r = floor(shortest / 4)

        xlim, ylim, zlim = shape
        #centres
        xc = floor(xlim / 2)
        yc = floor(ylim / 2)
        zc = floor(zlim / 2)

        x, y, z = np.mgrid[:xlim, :ylim, :zlim]
        inside = (x - xc)**2 + (y - yc)**2 + (z - zc)**2 < r**2

        s = np.where(
            inside,
            value * np.ones(shape),
            np.zeros(shape)
        )
        return cls(s) #TODO: check centre calculation

    @classmethod
    def cylinder(
        cls,
        shape: tuple[int],
        value: float = 1
    ):
        """Constructor to generate a cylinder.
        
        Generates a cylinder of susceptibility `value` in the centre of an 
        image with dimensions (x, y, z) in `shape`. The diameter of the 
        circular face, and the height of the cylinder are both half of the 
        shortest dimension.

        The cylinder is oriented with its height in the z direction.

        Parameters
        ----------
        shape : tuple[int, int, int]
            The resolution of the image in (x, y, z)
        
        value : float
            The value of the susceptibility inside the cylinder

        Returns
        -------
        Shapes
            A `Shapes` object containing a cylinder.

        See Also
        --------
        Shapes.sphere
            Constructor to generate a sphere.

        Shapes.cube
            Constructor to generate a cube.

        Notes
        -----
        To change the size/position/angle of the cylinder, apply
        the appropriate affine transformation.

        """
        shortest = min(shape)
        r = floor(shortest / 4)

        xlim, ylim, zlim = shape
        #centres
        xc = floor(xlim / 2)
        yc = floor(ylim / 2)
        zc = floor(zlim / 2)

        x, y, z = np.mgrid[:xlim, :ylim, :zlim]

        inside = ((x - xc)**2 + (y - yc)**2 < r ** 2) & (np.abs(z - zc)) < r

        s = np.where(
            inside,
            value * np.ones(shape),
            np.zeros(shape)
        )
        return cls(s)

    @classmethod
    def cube(
        cls,
        shape: tuple[int],
        value: float = 1
    ):
        """Constructor to generate a cube.

        Generates a cube of susceptibility `value` in the centre of an 
        image with dimensions (x, y, z) in `shape`. The side length of 
        the cube is half the shortest dimension.

        Parameters
        ----------
        shape : tuple[int, int, int]
            The resolution of the image in (x, y, z)
        
        value : float
            The value of the susceptibility inside the sphere

        Returns
        -------
        Shapes
            A `Shapes` object containing a cube.

        See Also
        --------
        Shapes.sphere
            Constructor to generate a sphere.

        Shapes.cylinder
            Constructor to generate a cylinder.

        Notes
        -----
        To change the size/position/angle of the cube, or to produce 
        a cuboid or parallelipiped, apply the appropriate affine 
        transformation.
        
        """
        shortest = min(shape)
        r = floor(shortest / 4)

        xlim, ylim, zlim = shape
        #centres
        xc = floor(xlim / 2)
        yc = floor(ylim / 2)
        zc = floor(zlim / 2)

        x, y, z = np.mgrid[:xlim, :ylim, :zlim]

        inside = (np.abs(x - xc)) < r & (np.abs(y - yc)) < r & (np.abs(z - zc)) < r

        s = np.where(
            inside,
            value * np.ones(shape),
            np.zeros(shape)
        )
        return cls(s)


class AffineTransform:
    def __init__(self, matrix:np.ndarray) -> None:
        """Performs affine transformations on susceptibility images.

        Class for applying affine transformations to `Shapes` objects. Provides 
        utilities to initialise affine transformations either from a set of 
        affine parameters (translation, rotation, scaling, shear) or randomly. 
        
        """
        self._matrix = matrix

    def __call__(self, object):
        return self.transform_shape(object)

    def __matmul__(self, other):
        if isinstance(other, AffineTransform):
            product = self.matrix @ other.matrix
            return AffineTransform(product)
        else:
            return self.matrix @ other

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: np.ndarray):
        m = np.array(new_matrix)

        #Checks
        if not np.issubdtype(m.dtype, np.number):
            raise ValueError("Matrices must contain numeric data.")
        if not m.shape == (4, 4):
            raise ValueError("The transformation must be a 4x4 matrix")
        if not np.all(m[3] == [0, 0, 0, 1]):
            raise ValueError("The last row of the matrix must be [0, 0, 0, 1]")

        self._matrix = m

    @matrix.deleter
    def matrix(self):
        del self._matrix

    @classmethod
    def from_params(
        cls, 
        params=[0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.]
        ):
        """Construct affine transformation matrix from affine parameters.

        Parameters
        ----------
        params : list[float]
            parameters for defining the construction of an affine transformation matrix
            in the format - 
                [tx, ty, tz, rx, ry, rz, sx, sy, sz, shxy, shxz, shyx, shyz, shzx, shzy]
            
            t<dim> - translation along axis <dim>
            r<dim> - rotation about axis <dim> in radians
            s<dim> - scaling along axis <dim>
            sh<dim1, dim2> - shearing

        Returns
        -------
        AffineTransform
            An object containing the matrix required for the specified affine transformation.

        """
        tx, ty, tz, rx, ry, rz, sx, sy, sz, shxy, shxz, shyx, shyz, shzx, shzy = params

        srx, sry, srz = np.sin((rx, ry, rz))
        crx, cry, crz = np.sin((rx, ry, rz))

        T_t = np.array(
            [
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0,  1]
            ]
        )
        T_rx = np.array(
            [
                [1,   0,    0, 0],
                [0, crx, -srx, 0],
                [0, srx,  crx, 0],
                [0,   0,    0, 1]
            ]
        )
        T_ry = np.array(
            [
                [cry, 0, -sry, 0],
                [0  , 1,    0, 0],
                [sry, 0,  cry, 0],
                [0  , 0,    0, 1]
            ]
        )
        T_rz = np.array(
            [
                [crz, -srz, 0, 0],
                [srz,  crz, 0, 0],
                [  0,    0, 1, 0],
                [  0,    0, 0, 1]
            ]
        )
        T_sc = np.array(
            [
                [sx,  0,  0, 0],
                [ 0, sy,  0, 0],
                [ 0,  0, sz, 0],
                [ 0,  0,  0, 1]
            ]
        )
        T_sh = np.array(
            [
                [   1, shyx, shzx, 0],
                [shxy,    1, shzy, 0],
                [shxz, shyz,    1, 0],
                [   0,    0,    0, 1]
            ]
        )

        T_pre = np.array(
            [
                [],
                [],
                [],
                []
            ]
        )

        T_post = np.array(
            [
                [],
                [],
                [],
                []
            ]
        )
        #TODO: rotate about the centre

        T_tot = T_t @ T_rx @ T_ry @ T_rz @ T_sh @ T_sc
        return cls(T_tot)

    @classmethod
    def from_random(cls, **kwargs):
        """Construct a random affine transformation matrix.

        Initialises the AffineTransform with random parameters as per the randomise 
        method.

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to the randomise method.

        Returns
        -------
        AffineTransform
            An affine transform with random parameters.

        See Also
        --------
        AffineTransform.randomise
            Change the parametrs of the affine transform to new, randomly selected, 
            parameters.
        
        """
        transform = cls(np.eye(4))
        transform.randomise(**kwargs)

        return transform

    def randomise(self, seed: int) -> None:
        """Randomise the parameters of the transform.
        
        """
        rng = np.random.default_rng(seed)
        raise NotImplementedError

    def transform_shape(self, shape: Shapes, **kwargs) -> Shapes:
        """Produce a `Shapes` obejct by transforming a shape with the affine transform.

        Parameters
        ----------
        shape: Shapes
            The distribution to which the transformation will be applied.

        **kwargs
            Keyword arguments passed to `utils.resample_with_deformation_field`.

        Returns
        -------
        Shapes
            A `Shapes` ojbect containing the transformed distribution.

        See Also
        --------
        utils.resample_with_deformation_field
            This function is used by this method to produce the transformed image. 
            See this function to find options which can be passed to control how the 
            interpolation is done when producing the transformed image.
        
        """
        def_field = deformation_field_from_affine_matrix(self.matrix, shape.shape)
        resampled_dist = resample_with_deformation_field(shape.dist, def_field, **kwargs)

        s = Shapes(resampled_dist)
        return s


class Distribution:
    def __init__(self, dist: np.ndarray) -> None:
        """A distribution of magnetic suscpetibility.
        
        """
        self._dist = dist

    @property
    def dist(self):
        return self._dist

    @classmethod
    def from_collection(cls, collection: list[Shapes]):
        total = sum(collection)
        return cls(total.dist)

    def save(self, filename):
        filepath = Path(filename)
        np.save(filepath, self.dist)
        raise NotImplementedError
