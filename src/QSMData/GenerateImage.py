from argparse import ArgumentParser

from .Susceptibility import Shapes, Distribution


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
    collection = []
    
    for _ in range(args.no_spheres):
        collection.append(Shapes.sphere())
    for _ in range(args.no_cylinders):
        collection.append(Shapes.cylinder())
    for _ in range(args.no_cuboids):
        collection.append(Shapes.cuboid())

    dist = Distribution.from_collection(collection)


def main():
    generate()


if __name__ == '__main__':
    main()