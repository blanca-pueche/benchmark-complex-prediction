#!/usr/bin/env python3

import argparse
import random
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select N unique IDs from an input file, "
            "optionally excluding IDs listed in a second file."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file containing comma-separated IDs."
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        required=True,
        help="Number of IDs to randomly select."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file where selected IDs will be saved, one per line."
    )

    parser.add_argument(
        "-e",
        "--exclude",
        default=None,
        help="Optional file containing IDs to exclude, one ID per line."
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling."
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Read IDs from input file
    with open(args.input, "r") as f:
        ids = [x.strip() for x in f.read().split(",") if x.strip()]

    # Remove duplicates while preserving order
    ids = list(dict.fromkeys(ids))

    # Read IDs to exclude
    excluded = set()

    if args.exclude:
        with open(args.exclude, "r") as f:
            excluded = {x.strip() for x in f if x.strip()}

        print(f"Loaded {len(excluded)} IDs to exclude.")

    # Remove excluded IDs
    available_ids = [x for x in ids if x not in excluded]

    if args.number > len(available_ids):
        sys.exit(
            f"Error: requested {args.number} IDs, but only "
            f"{len(available_ids)} IDs are available after exclusions."
        )

    # Randomly select IDs
    selected = random.sample(available_ids, args.number)

    # Write output
    with open(args.output, "w") as f:
        f.write("\n".join(selected))

    print(f"Selected {args.number} IDs from {len(available_ids)} available.")
    print(f"Excluded {len(ids) - len(available_ids)} IDs.")
    print(f"Saved to '{args.output}'.")


if __name__ == "__main__":
    main()