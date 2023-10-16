from argparse import ArgumentParser
import threading

from .Susceptibility import Shapes, AffineTransform, join_shapes


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
        '--filename',
        '-f',
        help="The filename to save the generated image to."
    )
    parser.add_argument(
        '--dir',
        '-d',
        help="The directory to save the geneated image to."
    )
    parser.add_argument(
        '--no_images',
        '-i',
        help="The number of images to produce.",
        type=int,
        default=1
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


def if_single_img(
    shape,
    seed,
    no_spheres,
    no_cuboids,
    no_cylinders,
    filename
):
    img = generate(
        shape=shape,
        seed=seed,
        no_spheres=no_spheres,
        no_cuboids=no_cuboids,
        no_cylinders=no_cylinders
    )

    if filename:
        img.save(filename)

    img.pad()
    phase = img.phase[
        shape[0]//2 : shape[0]//2 + shape[0],
        shape[1]//2 : shape[1]//2 + shape[1],
        shape[2]//2 : shape[2]//2 + shape[2]
    ]

    if filename:
        phase.save(filename + "_phase")


def if_multiple_imgs(
    no_images,
    shape,
    seed,
    no_spheres,
    no_cuboids,
    no_cylinders,
    filename
):
    imgs = []
    for _ in range(no_images):
        img = generate(
            shape=shape,
            seed=seed,
            no_spheres=no_spheres,
            no_cuboids=no_cuboids,
            no_cylinders=no_cylinders
        )
        imgs.append(img)
    img = join_shapes(imgs)

    if filename:
        img.save(filename)
    
    img.pad(
        (
            (0, 0),
            (shape[0]//2, shape[0]//2),
            (shape[1]//2, shape[1]//2),
            (shape[2]//2, shape[2]//2)
        )
    )
    phase = img.phase[
        :,
        shape[0]//2 : shape[0]//2 + shape[0],
        shape[1]//2 : shape[1]//2 + shape[1],
        shape[2]//2 : shape[2]//2 + shape[2]
    ]

    if filename:
        phase.save(filename + "_phase")


def main():
    args = parse_arguments()

    if args.no_images == 1:
        if_single_img(
            shape=args.shape,
            seed=args.seed,
            no_spheres=args.no_spheres,
            no_cuboids=args.no_cuboids,
            no_cylinders=args.no_cylinders,
            filename=args.filename
        )
    elif args.no_images > 1:
        if_multiple_imgs(
            no_images=args.no_images,
            shape=args.shape,
            seed=args.seed,
            no_spheres=args.no_spheres,
            no_cuboids=args.no_cuboids,
            no_cylinders=args.no_cylinders,
            filename=args.filename
        )
    else:
        raise ValueError("The number of images must be a positive integer.")


if __name__ == '__main__':
    main()
