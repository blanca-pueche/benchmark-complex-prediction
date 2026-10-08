#!/usr/bin/env python3

import argparse
import random
from pathlib import Path


def read_fasta_sequences(fasta_file):
    """Return all sequences from a FASTA file."""

    sequences = []
    current_sequence = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_sequence:
                    sequences.append("".join(current_sequence))
                    current_sequence = []
            else:
                current_sequence.append(line)

    if current_sequence:
        sequences.append("".join(current_sequence))

    return sequences


def contains_x(fasta_file):
    """Return True if any FASTA sequence contains X."""

    sequences = read_fasta_sequences(fasta_file)

    return any(
        "X" in sequence.upper()
        for sequence in sequences
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter FASTA files by removing files containing X, "
            "files already present in another folder, and optionally "
            "randomly selecting a fixed number of remaining files."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Folder containing FASTA files to filter."
    )

    parser.add_argument(
        "-e",
        "--exclude-folder",
        required=True,
        help=(
            "Folder whose filenames should be excluded "
            "from the input FASTA folder."
        )
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help=(
            "Number of FASTA files to keep randomly after filtering. "
            "If omitted, all remaining files are kept."
        ),
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection.",
    )

    args = parser.parse_args()

    if args.number is not None and args.number <= 0:
        parser.error("--number must be greater than 0.")

    fasta_folder = Path(args.input)
    other_folder = Path(args.exclude_folder)

    if not fasta_folder.is_dir():
        raise SystemExit(
            f"ERROR: FASTA folder does not exist: "
            f"{fasta_folder}"
        )

    if not other_folder.is_dir():
        raise SystemExit(
            f"ERROR: Comparison folder does not exist: "
            f"{other_folder}"
        )

    other_files = {
        file.name
        for file in other_folder.iterdir()
        if file.is_file()
    }

    fasta_files = sorted(
        file
        for file in fasta_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in {".fasta", ".fa", ".faa"}
    )

    print(f"Found {len(fasta_files)} FASTA files.")
    print(f"Found {len(other_files)} files in comparison folder.")
    print()

    deleted_x = []
    deleted_duplicate = []
    kept = []

    # --------------------------------------------------
    # First filter: duplicates and X residues
    # --------------------------------------------------

    for fasta_file in fasta_files:

        if fasta_file.name in other_files:

            print(
                f"[DELETE - duplicate] {fasta_file.name}"
            )

            fasta_file.unlink()
            deleted_duplicate.append(fasta_file.name)
            continue

        try:
            has_x = contains_x(fasta_file)

        except Exception as e:

            print(
                f"[ERROR] Could not read "
                f"{fasta_file.name}: {e}"
            )
            continue

        if has_x:

            print(
                f"[DELETE - X] {fasta_file.name}"
            )

            fasta_file.unlink()
            deleted_x.append(fasta_file.name)
            continue

        kept.append(fasta_file)

    # --------------------------------------------------
    # Second filter: random selection
    # --------------------------------------------------

    deleted_random = []

    if args.number is not None:

        if args.number > len(kept):
            parser.error(
                f"--number ({args.number}) is greater than "
                f"the number of remaining FASTA files "
                f"({len(kept)})."
            )

        if args.seed is not None:
            random.seed(args.seed)

        selected = set(
            random.sample(
                kept,
                args.number
            )
        )

        for fasta_file in kept:

            if fasta_file not in selected:

                print(
                    f"[DELETE - random] {fasta_file.name}"
                )

                fasta_file.unlink()
                deleted_random.append(
                    fasta_file.name
                )

        kept = [
            fasta_file
            for fasta_file in kept
            if fasta_file in selected
        ]

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"Initial FASTAs:       {len(fasta_files)}"
    )
    print(
        f"Deleted (X):           {len(deleted_x)}"
    )
    print(
        f"Deleted (duplicate):   {len(deleted_duplicate)}"
    )
    print(
        f"Deleted (random):      {len(deleted_random)}"
    )
    print(
        f"Remaining:             {len(kept)}"
    )
    print("=" * 60)

    if deleted_x:
        print("\nDeleted because of X:")
        for name in deleted_x:
            print(f"  {name}")

    if deleted_duplicate:
        print("\nDeleted because already in other folder:")
        for name in deleted_duplicate:
            print(f"  {name}")

    if deleted_random:
        print("\nDeleted by random selection:")
        for name in deleted_random:
            print(f"  {name}")

    if kept:
        print("\nFinal FASTAs:")
        for fasta_file in sorted(kept):
            print(f"  {fasta_file.name}")


if __name__ == "__main__":
    main()