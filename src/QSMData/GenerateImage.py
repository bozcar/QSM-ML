from argparse import ArgumentParser

from .Susceptibility import Shapes, AffineTransform


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
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--seed',
        help="The seed for the random number generator.",
        type=int,
        default=None
    )
    parser.add_argument(
        '--path',
        '-p',
        help="The path to save the generated image to."
    )

    arguments = parser.parse_args()
    return arguments


def generate(
    shape: list[int],
    seed: int,
    no_spheres: int,
    no_cuboids: int,
    no_cylinders: int
) -> Shapes:
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
    transform = AffineTransform.from_random(seed=seed, shape=shape)

    img = Shapes.empty(shape)

    sphere = Shapes.sphere(shape)
    cube = Shapes.cube(shape)
    cylinder = Shapes.cylinder(shape)

    for _ in range(no_spheres):
        transform.randomise(shape = shape)
        img += transform(sphere)
    for _ in range(no_cylinders):
        transform.randomise(shape = shape)
        img += transform(cylinder)
    for _ in range(no_cuboids):
        transform.randomise(shape = shape)
        img += transform(cube)

    return img


def main():
    args = parse_arguments()
    path = args.path

    img = generate(
        shape=args.shape,
        seed=args.seed,
        no_spheres=args.no_spheres,
        no_cuboids=args.no_cuboids,
        no_cylinders=args.no_cylinders
    )
    if path:
        img.save(path)

    img.pad()
    phase = img.phase[
        args.shape[0]//2 : args.shape[0]//2 + args.shape[0],
        args.shape[1]//2 : args.shape[1]//2 + args.shape[1],
        args.shape[2]//2 : args.shape[2]//2 + args.shape[2]
    ]
    if path:
        phase.save(path + "_phase")

    img.display_slice(args.shape[2]//2)
    phase.display_slice(args.shape[2]//2)


if __name__ == '__main__':
    main()
