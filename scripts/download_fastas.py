#!/usr/bin/env python3

import argparse
from pathlib import Path
import requests


def download_fasta(pdb_id, out_dir, overwrite=False):
    """Download the FASTA file for a PDB entry from RCSB."""

    pdb_id = pdb_id.strip().upper()

    if not pdb_id:
        return

    outfile = out_dir / f"{pdb_id}.fasta"

    if outfile.exists() and not overwrite:
        print(f"[SKIP] {pdb_id} already exists.")
        return

    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        if not response.text.startswith(">"):
            print(f"[ERROR] {pdb_id}: No FASTA returned.")
            return

        outfile.write_text(response.text)

        print(f"[OK] Downloaded {pdb_id}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {pdb_id}: {e}")


def read_pdb_ids(txt_file):
    """Read unique PDB IDs from a text file."""

    with open(txt_file) as f:
        ids = {
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        }

    return sorted(ids)


def main():

    parser = argparse.ArgumentParser(
        description="Download FASTA files from the RCSB PDB."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Text file containing one PDB ID per line."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory where FASTA files will be saved."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FASTA files."
    )

    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdb_ids = read_pdb_ids(args.input)

    print(
        f"Found {len(pdb_ids)} PDB IDs.\n"
    )

    for pdb_id in pdb_ids:

        download_fasta(
            pdb_id,
            out_dir,
            overwrite=args.overwrite
        )

    print("\nDone.")


if __name__ == "__main__":
    main()