from argparse import ArgumentParser

from .Susceptibility import Shapes, Distribution, AffineTransform


def parse_arguments():
    parser = ArgumentParser(
        description="Create a susceptibility distribution and its associated phase distribution."
    )

    parser.add_argument(
        '--no_spheres',
        '-ns',
        help="The number of spheres in the susceptibility distribution.",
        type=int
    )
    parser.add_argument(
        '--no_cylinders',
        '-ncy',
        help="The number of cylinders in the susceptibility distribution.",
        type=int
    )
    parser.add_argument(
        '--no_cuboids',
        '-ncb',
        help="The number of cuboids in the susceptibility distribution.",
        type=int
    )
    parser.add_argument(
        '--shape',
        '-s',
        help="The number of pixels in each image dimension",
        type=list[int] #TODO: test this, I'm sure it will break.
    )

    arguments = parser.parse_args()
    return arguments


def generate(args: ArgumentParser):
    """Generate a susceptibility distribution and its associated phase image.

    Parameters
    ----------
    no_spheres : int
        the number of spheres in the susceptibility distribution
    
    no_cylinders : int
        the number of cylinders in the susceptibility distribution

    no_cuboids : int
        the number of cuboids in the susceptibility distribution
    
    """
    transform = AffineTransform.from_random()

    img = Shapes.empty((64, 64, 64))

    sphere = Shapes.sphere((64, 64, 64))
    cube = Shapes.cube((64, 64, 64))
    cylinder = Shapes.cylinder((64, 64, 64))

    for _ in range(args.no_spheres):
        transform.randomise()
        img += transform(sphere)
    for _ in range(args.no_cylinders):
        transform.randomise()
        img += transform(cylinder)
    for _ in range(args.no_cuboids):
        transform.randomise()
        img += transform(cube)

    dist = Distribution.from_collection(img)


def main():
    generate()


if __name__ == '__main__':
    main()