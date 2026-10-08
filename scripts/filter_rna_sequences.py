#!/usr/bin/env python3

import argparse
import requests
import time


RNA_LETTERS = set("AUGCIXN")


def is_rna(seq):
    """Return True if sequence looks like RNA."""
    seq = seq.upper().replace("\n", "")
    return len(seq) > 0 and set(seq) <= RNA_LETTERS


def download_fasta(pdb_id):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    r = requests.get(url, timeout=20)

    if r.status_code != 200:
        return None

    return r.text


def parse_fasta(text):
    records = []
    header = None
    seq = []

    for line in text.splitlines():

        if line.startswith(">"):

            if header is not None:
                records.append(
                    (header, "".join(seq))
                )

            header = line[1:]
            seq = []

        else:

            seq.append(line.strip())

    if header is not None:
        records.append(
            (header, "".join(seq))
        )

    return records


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter PDB IDs by checking whether all RNA sequences "
            "in each structure have at least a minimum length."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Input file containing PDB IDs. "
            "IDs may be comma-separated and/or one per line."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "Output file where accepted PDB IDs will be saved, "
            "one ID per line."
        )
    )

    args = parser.parse_args()

    valid = []

    # Read IDs (supports comma-separated and/or one per line)
    with open(args.input) as f:
        text = f.read()

    pdb_ids = [
        pdb.strip().upper()
        for pdb in text.replace("\n", ",").split(",")
        if pdb.strip()
    ]

    for pdb in pdb_ids:

        print(pdb)

        fasta = download_fasta(pdb)

        if fasta is None:

            print("  failed")
            continue

        records = parse_fasta(fasta)

        rna_lengths = []

        for header, seq in records:

            if is_rna(seq):
                rna_lengths.append(len(seq))

        if not rna_lengths:

            print("  no RNA?")
            continue

        if all(length >= 7 for length in rna_lengths):

            valid.append(pdb)

            print(
                f"  OK ({rna_lengths})"
            )

        else:

            print(
                f"  rejected ({rna_lengths})"
            )

        time.sleep(0.2)

    with open(args.output, "w") as f:

        for pdb in valid:
            f.write(pdb + "\n")

    print(
        f"\nKept {len(valid)} structures."
    )


if __name__ == "__main__":
    main()