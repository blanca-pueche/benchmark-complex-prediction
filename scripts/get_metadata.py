#!/usr/bin/env python3

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


# ============================================================
# RCSB API
# ============================================================

def get_rcsb_entry(pdb_id):
    """
    Retrieve entry-level information from RCSB PDB.
    """

    url = (
        "https://data.rcsb.org/rest/v1/core/entry/"
        + pdb_id.lower()
    )

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())

    except Exception as e:
        print(
            f"  WARNING: Could not retrieve RCSB data "
            f"for {pdb_id}: {e}"
        )
        return None


def get_rcsb_polymer_entities(pdb_id):
    """
    Retrieve all polymer entities for a PDB entry.

    Polymer entities represent unique polymer sequences.
    Each entity can correspond to one or more physical chains.
    """

    entities = []
    entity_id = 1

    while True:

        url = (
            "https://data.rcsb.org/rest/v1/core/polymer_entity/"
            f"{pdb_id.lower()}/{entity_id}"
        )

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read())

        except Exception:
            break

        entities.append(data)
        entity_id += 1

    return entities


def get_structure_chain_information(pdb_id):
    """
    Obtain the actual chain instances in the PDB structure.

    Returns:
        chain_ids
        chain_types
        chain_lengths
        total_structure_residues
    """

    entities = get_rcsb_polymer_entities(pdb_id)

    chain_ids = []
    chain_types = []
    chain_lengths = []

    for entity in entities:

        identifiers = entity.get(
            "rcsb_polymer_entity_container_identifiers",
            {}
        )

        # Actual chain IDs in the structure
        auth_asym_ids = identifiers.get(
            "auth_asym_ids",
            []
        )

        # Sequence information
        sequence = entity.get(
            "entity_poly",
            {}
        ).get(
            "pdbx_seq_one_letter_code_can",
            ""
        )

        sequence = sequence.replace(
            "\n",
            ""
        ).replace(
            " ",
            ""
        )

        length = len(sequence)

        # Determine polymer type
        polymer_type = (
            entity.get("entity_poly", {})
            .get("type", "")
        )

        if "polypeptide" in polymer_type.lower():

            chain_type = "Protein"

        elif "deoxyribonucleotide" in polymer_type.lower():

            chain_type = "DNA"

        elif "ribonucleotide" in polymer_type.lower():

            chain_type = "RNA"

        else:

            chain_type = polymer_type or "Unknown"

        # One entity can represent multiple identical chains
        for chain_id in auth_asym_ids:

            chain_ids.append(chain_id)
            chain_types.append(chain_type)
            chain_lengths.append(str(length))

    total_residues = sum(
        int(length)
        for length in chain_lengths
        if length.isdigit()
    )

    return {
        "structure_chain_count": len(chain_ids),
        "structure_chain_ids": ";".join(chain_ids),
        "structure_chain_types": ";".join(chain_types),
        "structure_chain_lengths": ";".join(chain_lengths),
        "structure_residues": total_residues,
    }


# ============================================================
# FASTA parsing
# ============================================================

PROTEIN = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
DNA = set("ATCGN")
RNA = set("AUCGN")


def classify_sequence(seq):
    """
    Classify a sequence as Protein, DNA, RNA, NucleicAcid
    or Unknown.
    """

    s = set(seq.upper())

    if s <= DNA:
        return "DNA"

    if s <= RNA:
        return "RNA"

    if s <= PROTEIN:
        return "Protein"

    if s <= (DNA | RNA):
        return "NucleicAcid"

    return "Unknown"


def parse_fasta(fasta):
    """
    Read a FASTA file.

    Each FASTA entry represents one unique sequence.
    """

    sequences = []
    current = None

    with open(fasta, encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if current is not None:
                    sequences.append(current)

                header = line[1:]

                current = {
                    "header": header,
                    "id": header.split()[0],
                    "seq": ""
                }

            else:

                if current is not None:
                    current["seq"] += line

    if current is not None:
        sequences.append(current)

    return sequences


# ============================================================
# Extract RCSB metadata
# ============================================================

def extract_rcsb_metadata(data):

    if not data:
        return {}

    result = {}

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    result["rcsb_title"] = (
        data.get("struct", {})
        .get("title", "")
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    accession_info = data.get(
        "rcsb_accession_info",
        {}
    )

    result["deposit_date"] = accession_info.get(
        "deposit_date",
        ""
    )

    result["initial_release_date"] = accession_info.get(
        "initial_release_date",
        ""
    )

    result["revision_date"] = accession_info.get(
        "major_revision_date",
        ""
    )

    # --------------------------------------------------------
    # Experimental method
    # --------------------------------------------------------

    methods = []

    for experiment in data.get("exptl", []):

        method = experiment.get("method")

        if method and method not in methods:
            methods.append(method)

    result["experimental_method"] = ";".join(methods)

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    resolution = (
        data.get("rcsb_entry_info", {})
        .get("resolution_combined", [])
    )

    if resolution:
        result["resolution_A"] = min(resolution)
    else:
        result["resolution_A"] = ""

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    entry_info = data.get(
        "rcsb_entry_info",
        {}
    )

    result["polymer_entity_count"] = entry_info.get(
        "polymer_entity_count",
        ""
    )

    result["polymer_count"] = entry_info.get(
        "polymer_count",
        ""
    )

    result["assembly_count"] = entry_info.get(
        "assembly_count",
        ""
    )

    result["molecular_weight_Da"] = entry_info.get(
        "molecular_weight",
        ""
    )

    result["experimental_method_count"] = entry_info.get(
        "experimental_method_count",
        ""
    )

    # --------------------------------------------------------
    # Citation / DOI
    # --------------------------------------------------------

    citations = data.get(
        "citation",
        []
    )

    if citations:

        citation = citations[0]

        result["journal"] = citation.get(
            "journal_abbrev",
            ""
        )

        result["year"] = citation.get(
            "year",
            ""
        )

        result["doi"] = citation.get(
            "pdbx_database_id_doi",
            ""
        )

    else:

        result["journal"] = ""
        result["year"] = ""
        result["doi"] = ""

    return result


# ============================================================
# CSV fields
# ============================================================

fieldnames = [

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    "category",
    "pdb",
    "fasta_file",

    # --------------------------------------------------------
    # FASTA / unique sequences
    # --------------------------------------------------------

    "sequence_count",
    "sequence_ids",
    "sequence_types",
    "sequence_lengths",

    "protein_sequences",
    "dna_sequences",
    "rna_sequences",

    "unique_sequence_residues",

    # --------------------------------------------------------
    # Actual structure chains
    # --------------------------------------------------------

    "structure_chain_count",
    "structure_chain_ids",
    "structure_chain_types",
    "structure_chain_lengths",
    "structure_residues",

    # --------------------------------------------------------
    # RCSB entry information
    # --------------------------------------------------------

    "rcsb_title",

    "experimental_method",
    "experimental_method_count",
    "resolution_A",

    "deposit_date",
    "initial_release_date",
    "revision_date",

    "polymer_entity_count",
    "polymer_count",
    "assembly_count",

    "molecular_weight_Da",

    # --------------------------------------------------------
    # Publication
    # --------------------------------------------------------

    "journal",
    "year",
    "doi",
]


# ============================================================
# Read existing PDB IDs
# ============================================================

def get_existing_pdbs(output):
    """
    Return PDB IDs already present in the CSV.

    Existing rows are never modified.
    """

    existing = set()

    output_path = Path(output)

    if not output_path.exists():
        return existing

    with open(
        output_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            pdb_id = row.get(
                "pdb",
                ""
            ).strip().upper()

            if pdb_id:
                existing.add(pdb_id)

    return existing


# ============================================================
# Find FASTA files
# ============================================================

def get_fasta_files(root):
    """
    Find FASTA files inside category subdirectories.
    """

    fasta_files = []

    root = Path(root)

    if not root.exists():

        print(
            f"ERROR: Directory '{root}' does not exist."
        )

        return fasta_files

    for category in sorted(root.iterdir()):

        if not category.is_dir():
            continue

        for fasta in sorted(category.iterdir()):

            if fasta.suffix.lower() in [
                ".fa",
                ".fasta",
                ".faa"
            ]:

                fasta_files.append(
                    (category.name, fasta)
                )

    return fasta_files


# ============================================================
# Process one FASTA
# ============================================================

def process_fasta(category, fasta):

    # --------------------------------------------------------
    # PDB ID
    # --------------------------------------------------------

    pdb_id = fasta.stem[:4].upper()

    # --------------------------------------------------------
    # FASTA / unique sequences
    # --------------------------------------------------------

    sequences = parse_fasta(fasta)

    unique_sequence_residues = 0

    protein_sequences = 0
    dna_sequences = 0
    rna_sequences = 0

    sequence_lengths = []
    sequence_types = []
    sequence_ids = []

    for sequence in sequences:

        seq = sequence["seq"]

        length = len(seq)

        sequence_type = classify_sequence(seq)

        unique_sequence_residues += length

        sequence_lengths.append(
            str(length)
        )

        sequence_types.append(
            sequence_type
        )

        sequence_ids.append(
            sequence["id"]
        )

        if sequence_type == "Protein":

            protein_sequences += 1

        elif sequence_type == "DNA":

            dna_sequences += 1

        elif sequence_type == "RNA":

            rna_sequences += 1

    # --------------------------------------------------------
    # RCSB entry
    # --------------------------------------------------------

    print(
        "  Querying RCSB entry information..."
    )

    rcsb = get_rcsb_entry(pdb_id)

    if rcsb is None:

        print(
            f"  WARNING: Could not retrieve entry "
            f"information for {pdb_id}"
        )

    entry_metadata = extract_rcsb_metadata(
        rcsb
    )

    # --------------------------------------------------------
    # Actual structure chains
    # --------------------------------------------------------

    print(
        "  Querying RCSB chain information..."
    )

    structure_metadata = get_structure_chain_information(
        pdb_id
    )

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    row = {

        # Benchmark
        "category": category,
        "pdb": pdb_id,
        "fasta_file": fasta.name,

        # FASTA / unique sequences
        "sequence_count": len(sequences),
        "sequence_ids": ";".join(sequence_ids),
        "sequence_types": ";".join(sequence_types),
        "sequence_lengths": ";".join(sequence_lengths),

        "protein_sequences": protein_sequences,
        "dna_sequences": dna_sequences,
        "rna_sequences": rna_sequences,

        "unique_sequence_residues": (
            unique_sequence_residues
        ),
    }

    # Add RCSB entry metadata
    row.update(entry_metadata)

    # Add actual structure chain metadata
    row.update(structure_metadata)

    return row


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect FASTA and RCSB metadata for structures "
            "organized in category subdirectories."
        )
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help=(
            "Directory containing one subdirectory per category, "
            "with FASTA files inside each category directory."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file for the collected metadata."
    )

    args = parser.parse_args()

    root = Path(args.input_dir)
    output = Path(args.output)

    print("=" * 70)
    print("Benchmark metadata collection")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Existing structures
    # --------------------------------------------------------

    existing_pdbs = get_existing_pdbs(output)

    if existing_pdbs:

        print(
            f"Found {len(existing_pdbs)} existing PDB IDs "
            f"in {output}"
        )

    else:

        if output.exists():

            print(
                f"{output} exists but contains no PDB IDs."
            )

        else:

            print(
                f"{output} does not exist. "
                f"It will be created."
            )

    print()

    # --------------------------------------------------------
    # Find FASTAs
    # --------------------------------------------------------

    fasta_files = get_fasta_files(root)

    print(
        f"Found {len(fasta_files)} FASTA files "
        f"inside {root}/"
    )

    print()

    # --------------------------------------------------------
    # Select only new structures
    # --------------------------------------------------------

    new_fasta_files = []

    for category, fasta in fasta_files:

        pdb_id = fasta.stem[:4].upper()

        if pdb_id in existing_pdbs:

            print(
                f"[SKIP] {pdb_id} already exists "
                f"in {output}"
            )

            continue

        new_fasta_files.append(
            (category, fasta)
        )

    print()

    print(
        f"{len(new_fasta_files)} new structures "
        f"to process."
    )

    print()

    if not new_fasta_files:

        print("Nothing new to add.")
        return

    # --------------------------------------------------------
    # Open CSV in APPEND mode
    # --------------------------------------------------------

    file_exists = output.exists()

    # Create parent directory if necessary
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        # Header only for a new/empty file
        if (
            not file_exists
            or output.stat().st_size == 0
        ):

            writer.writeheader()

        # ----------------------------------------------------
        # Process structures
        # ----------------------------------------------------

        added = 0

        for index, (category, fasta) in enumerate(
            new_fasta_files,
            1
        ):

            print(
                f"[{index}/{len(new_fasta_files)}] "
                f"{category}/{fasta.name}"
            )

            try:

                row = process_fasta(
                    category,
                    fasta
                )

                writer.writerow(row)

                # Immediately save the row
                f.flush()

                pdb_id = row["pdb"]

                existing_pdbs.add(
                    pdb_id
                )

                added += 1

                print(
                    f"  [OK] Added {pdb_id}"
                )

            except Exception as e:

                print(
                    f"  [ERROR] Could not process "
                    f"{fasta.name}: {e}"
                )

            # Avoid hammering RCSB
            time.sleep(0.2)

            print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("Done!")
    print("=" * 70)

    print(
        f"Added: {added} new structures"
    )

    print(
        f"Existing structures left untouched: "
        f"{len(existing_pdbs) - added}"
    )

    print(
        f"Output: {output}"
    )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()