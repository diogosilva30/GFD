"""
Argument parser for `main.py`
"""
import argparse


def get_parser():
    # Create a parser
    parser = argparse.ArgumentParser(
        description="Golden flavescence detection using HSI"
    )
    parser.add_argument(
        "--bands",
        dest="bands",
        type=int,
        default=64,
        help="Number of bands to keep.",
    )
    parser.add_argument(
        "--patch-size",
        dest="patch_size",
        type=int,
        default=64,
        help="Size of patches.",
    )

    parser.add_argument(
        "--force-recreate",
        dest="force_recreate",
        default=False,
        action="store_true",
        help="Whether to (re-)clean and (re-)prepare the dataset. (Takes a lot of time!)",
    )
    parser.add_argument(
        "--no-force-recreate", dest="force_recreate", action="store_false"
    )

    parser.add_argument(
        "--baseline",
        dest="baseline",
        default=False,
        action="store_true",
        help="Whether to create a baseline model.",
    )
    parser.add_argument("--no-baseline", dest="baseline", action="store_false")

    parser.add_argument(
        "--allow-cpu",
        dest="allow_cpu",
        default=False,
        action="store_true",
        help="Whether to allow the code to run on CPU (Without GPU).",
    )
    parser.add_argument("--no-allow-cpu", dest="allow_cpu", action="store_false")

    return parser