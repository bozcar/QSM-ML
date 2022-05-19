import numpy as np

from QSMData.Susceptibility import AffineTransform


def test_init_from_params():
    no_params = AffineTransform.from_params()
    only_translation = AffineTransform.from_params((1, -1, 0.3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    only_rotation = AffineTransform.from_params((0, 0, 0, 1, -0.2, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    only_scaling = AffineTransform.from_params((0, 0, 0, 0, 0, 0, 2, -3, 0.7, 0, 0, 0, 0, 0, 0))
    only_shearing = AffineTransform.from_params((0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, -2, 0.7, -0.3, 6.1))

    assert no_params.matrix == np.eye(4)


def test_random_init():
    pass


def main():
    test_init_from_params()


if __name__ == '__main__':
    main()