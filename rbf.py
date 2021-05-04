import numpy as np
import argparse


find_fi = lambda x, y, sigma : np.float64(np.power(np.e, -(.8326 * np.abs(x - y) / sigma)**2))


def find_weights(random_nums, iterations, sigma=1):
    A = np.linalg.inv(np.array([[find_fi(fi, fi_2, sigma)
                                 for fi_2 in random_nums] for fi in random_nums], dtype=np.float64))
    return np.dot(A, np.array([np.sin(x)
                               for x in random_nums], dtype=np.float64).T)


def find_result(x, weights, random_nums, sigma=1):
    approx = np.float64(np.dot(np.array(
        [find_fi(x, y, sigma) for y in random_nums], dtype=np.float64), weights.T))

    print(f'Estimated value of sin({x}) = {approx}')
    print(f'Actual value of sin({x}) = {np.float64(np.sin(x))}', end='\n\n')
    print(f'Error = {np.abs(np.float64(np.sin(x))-approx)}')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--sigma",
        nargs='?',
        help="σ parameter value",
        const=1,
        type=float,
        default=1)
    parser.add_argument(
        "iterations",
        type=int,
        help="specify number of instances to train the network")
    parser.add_argument("num", type=float, help="value to test")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("Num. of iterations must be > 0")

    rng = np.random.default_rng()

    random_nums = rng.uniform(high=3 * np.pi, size=args.iterations)
    weights = find_weights(random_nums, args.iterations, sigma=args.sigma)
    find_result(args.num, weights, random_nums, sigma=args.sigma)


if __name__ == '__main__':
    main()
