#!/usr/bin/env python3

import argparse
import os
import json
import csv
import re
from datetime import datetime

from pyworkflow.project import Manager


FIELDS = [
    "category",
    "pdb_id",
    "predictor",

    "execution_time",

    "lddt",
    "bb_lddt",
    "tm_score",
    "rmsd",

    "qs_global",
    "qs_best",

    "dockq_ave",
    "dockq_wave",
    "dockq_ave_full",
    "dockq_wave_full",

    "oligo_gdtts",
    "oligo_gdtha",

    "num_clashes",
    "num_bad_bonds",
    "num_bad_angles",

    "status"
]


START_RE = re.compile(
    r"STARTED.*?time\s+"
    r"(\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}\.\d+)"
)


FINISH_RE = re.compile(
    r"FINISHED.*?time\s+"
    r"(\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}\.\d+)"
)


EXPECTED_PREDICTORS = [
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold",
    "AF3",
]


def getExecutionTime(prot):
    """Return total protocol execution time as 'M' SS.ss''."""

    stdout_file = os.path.join(
        prot.getWorkingDir(),
        "logs",
        "run.stdout"
    )

    if not os.path.exists(stdout_file):
        return None

    with open(stdout_file, errors="ignore") as f:
        text = f.read()

    starts = START_RE.findall(text)
    finishes = FINISH_RE.findall(text)

    if not starts or not finishes:
        return None

    start_time = datetime.strptime(
        starts[0],
        "%Y-%m-%d %H:%M:%S.%f"
    )

    finish_time = datetime.strptime(
        finishes[-1],
        "%Y-%m-%d %H:%M:%S.%f"
    )

    if finish_time < start_time:
        return None

    total_seconds = (
        finish_time - start_time
    ).total_seconds()

    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    return f"{minutes}' {seconds:.2f}''"


def getPredictorProtocol(project, predictor, pdb_id):

    labels = {
        "Boltz": f"Boltz {pdb_id}",
        "Chai": f"Chai {pdb_id}",
        "Protenix": f"Protenix {pdb_id}",
        "IntelliFold": f"IntelliFold {pdb_id}",
        "AF3": f"Import AF3 {pdb_id}",
    }

    expected_label = labels.get(predictor)

    if expected_label is None:
        return None

    for prot in project.getRuns():

        if prot.getObjLabel() == expected_label:
            return prot

    return None


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract OpenStructure comparison results and predictor "
            "execution times from Scipion benchmark projects."
        )
    )

    parser.add_argument(
        "-i",
        "--projects-dir",
        required=True,
        help=(
            "Directory containing the Scipion projects "
            "to process."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file."
    )

    args = parser.parse_args()

    projects_dir = os.path.abspath(
        args.projects_dir
    )

    output_csv = os.path.abspath(
        args.output
    )

    if not os.path.isdir(projects_dir):
        parser.error(
            f"Projects directory does not exist: "
            f"{projects_dir}"
        )

    manager = Manager()

    rows = []

    for project_name in sorted(
        os.listdir(projects_dir)
    ):

        if not project_name.startswith(
            "BioFoldBenchmark_"
        ):
            continue

        print(
            f"\nScanning {project_name}"
        )

        project = manager.loadProject(
            project_name
        )

        parts = project_name.split("_")

        category = parts[1]
        pdb_id = parts[2]

        for predictor in EXPECTED_PREDICTORS:

            prot = None

            for p in project.getRuns():

                if (
                    p.getClassName()
                    == "ProtCompareStructures"
                    and p.getObjLabel()
                    == (
                        f"OpenStructure "
                        f"{predictor} "
                        f"{pdb_id}"
                    )
                ):

                    prot = p
                    break

            # ---------------------------------------------------
            # Predictor execution time
            # ---------------------------------------------------

            if predictor == "AF3":

                execution_time = "N/A"

            else:

                prediction_prot = getPredictorProtocol(
                    project,
                    predictor,
                    pdb_id
                )

                execution_time = (
                    getExecutionTime(
                        prediction_prot
                    )
                    if prediction_prot is not None
                    else None
                )

            # ---------------------------------------------------
            # If comparison protocol does not exist,
            # write empty row
            # ---------------------------------------------------

            if prot is None:

                rows.append({

                    "category": category,
                    "pdb_id": pdb_id,
                    "predictor": predictor,

                    "execution_time": execution_time,

                    "lddt": "",
                    "bb_lddt": "",
                    "tm_score": "",
                    "rmsd": "",

                    "qs_global": "",
                    "qs_best": "",

                    "dockq_ave": "",
                    "dockq_wave": "",
                    "dockq_ave_full": "",
                    "dockq_wave_full": "",

                    "oligo_gdtts": "",
                    "oligo_gdtha": "",

                    "num_clashes": "",
                    "num_bad_bonds": "",
                    "num_bad_angles": "",

                    "status": "FAIL"
                })

                continue

            # ---------------------------------------------------
            # Read comparison results
            # ---------------------------------------------------

            json_file = os.path.join(
                prot.getWorkingDir(),
                "compare_structures.json"
            )

            if not os.path.exists(json_file):

                json_file = os.path.join(
                    prot.getWorkingDir(),
                    "extra",
                    "compare_structures.json"
                )

            if not os.path.exists(json_file):

                rows.append({

                    "category": category,
                    "pdb_id": pdb_id,
                    "predictor": predictor,

                    "execution_time": execution_time,

                    "lddt": "",
                    "bb_lddt": "",
                    "tm_score": "",
                    "rmsd": "",

                    "qs_global": "",
                    "qs_best": "",

                    "dockq_ave": "",
                    "dockq_wave": "",
                    "dockq_ave_full": "",
                    "dockq_wave_full": "",

                    "oligo_gdtts": "",
                    "oligo_gdtha": "",

                    "num_clashes": "",
                    "num_bad_bonds": "",
                    "num_bad_angles": "",

                    "status": "FAIL"
                })

                continue

            with open(json_file) as f:
                data = json.load(f)

            rows.append({

                "category": category,
                "pdb_id": pdb_id,
                "predictor": predictor,

                "execution_time": execution_time,

                "lddt": data.get("lddt"),
                "bb_lddt": data.get("bb_lddt"),
                "tm_score": data.get("tm_score"),
                "rmsd": data.get("rmsd"),

                "qs_global": data.get("qs_global"),
                "qs_best": data.get("qs_best"),

                "dockq_ave": data.get("dockq_ave"),
                "dockq_wave": data.get("dockq_wave"),
                "dockq_ave_full": data.get(
                    "dockq_ave_full"
                ),
                "dockq_wave_full": data.get(
                    "dockq_wave_full"
                ),

                "oligo_gdtts": data.get(
                    "oligo_gdtts"
                ),
                "oligo_gdtha": data.get(
                    "oligo_gdtha"
                ),

                "num_clashes": len(
                    data.get(
                        "model_clashes",
                        []
                    )
                ),

                "num_bad_bonds": len(
                    data.get(
                        "model_bad_bonds",
                        []
                    )
                ),

                "num_bad_angles": len(
                    data.get(
                        "model_bad_angles",
                        []
                    )
                ),

                "status": "SUCCESS"
            })

    rows.sort(
        key=lambda r: (
            r["category"],
            r["pdb_id"],
            r["predictor"]
        )
    )

    output_parent = os.path.dirname(
        output_csv
    )

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True
        )

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print(
        f"Wrote {len(rows)} comparisons"
    )
    print(
        f"CSV written to: {output_csv}"
    )


if __name__ == "__main__":
    main()