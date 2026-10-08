#!/usr/bin/env python3

"""
01_prepare_dataset.py

Prepare the OpenStructure benchmark results for statistical analysis.

Inputs
------
- OpenStructure results CSV
- Benchmark metadata CSV

Expected predictors
-------------------
- AF3
- Boltz
- Chai
- Protenix
- IntelliFold

The script:
1. Loads OpenStructure results.
2. Loads benchmark metadata.
3. Ensures every structure has one row for every predictor.
4. Adds missing predictor rows as FAIL.
5. Keeps failed predictions with empty structural metrics.
6. Parses execution time into seconds.
7. Standardizes category names.
8. Adds structure-level success information.
9. Merges RCSB metadata.
10. Generates summary tables.

Output directory
----------------
Specified with --output-dir (default: analysis/)
"""

import os
import re
import argparse

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTORS = [
    "AF3",
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold",
]

METRICS = [
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
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def standardize_category(category):
    """
    Convert different category naming conventions into a common format.
    """

    if pd.isna(category):
        return np.nan

    value = str(category).strip()

    normalized = (
        value.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    mapping = {
        "proteinprotein": "Protein-Protein",
        "proteinantibody": "Protein-Antibody",
        "antibody": "Protein-Antibody",
        "proteindna": "Protein-DNA",
        "dna": "Protein-DNA",
        "proteinrna": "Protein-RNA",
        "rna": "Protein-RNA",
        "proteinpeptide": "Protein-Peptide",
        "peptide": "Protein-Peptide",
    }

    return mapping.get(normalized, value)


def parse_execution_time(value):
    """
    Convert execution-time strings into seconds.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.upper() in [
        "N/A",
        "NA",
        "",
        "NONE",
        "NAN",
    ]:
        return np.nan

    match = re.match(
        r"^\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*''\s*$",
        value
    )

    if match:

        minutes = float(match.group(1))
        seconds = float(match.group(2))

        return minutes * 60 + seconds

    try:
        return float(value)

    except ValueError:

        return np.nan


def clean_numeric_columns(df):
    """
    Convert metric columns to numeric values.
    """

    for column in METRICS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare OpenStructure benchmark results "
            "for statistical analysis."
        )
    )

    parser.add_argument(
        "-r",
        "--results",
        required=True,
        help=(
            "OpenStructure results CSV file."
        )
    )

    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        help=(
            "Benchmark metadata CSV file."
        )
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis",
        help=(
            "Output directory for analysis files "
            "(default: analysis)."
        )
    )

    args = parser.parse_args()

    openstructure_file = os.path.abspath(
        args.results
    )

    metadata_file = os.path.abspath(
        args.metadata
    )

    output_dir = os.path.abspath(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not os.path.exists(openstructure_file):

        raise FileNotFoundError(
            f"Could not find {openstructure_file}"
        )

    if not os.path.exists(metadata_file):

        raise FileNotFoundError(
            f"Could not find {metadata_file}"
        )

    results = pd.read_csv(
        openstructure_file
    )

    metadata = pd.read_csv(
        metadata_file
    )

    print(
        f"OpenStructure rows: {len(results)}"
    )

    print(
        f"Metadata rows:       {len(metadata)}"
    )


    # ========================================================
    # BASIC CLEANING
    # ========================================================

    results.columns = (
        results.columns.str.strip()
    )

    metadata.columns = (
        metadata.columns.str.strip()
    )

    results["pdb_id"] = (
        results["pdb_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    metadata["pdb_id"] = (
        metadata["pdb"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if "resolution_A" in metadata.columns:

        metadata["resolution"] = pd.to_numeric(
            metadata["resolution_A"],
            errors="coerce"
        )

    if "category" in results.columns:

        results["category"] = (
            results["category"]
            .apply(standardize_category)
        )

    if "category" in metadata.columns:

        metadata["category"] = (
            metadata["category"]
            .apply(standardize_category)
        )

    results["predictor"] = (
        results["predictor"]
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # STANDARDIZE PREDICTOR NAMES
    # ========================================================

    predictor_mapping = {

        "AF3": "AF3",
        "Import AF3": "AF3",
        "AlphaFold3": "AF3",
        "AlphaFold 3": "AF3",

        "Boltz": "Boltz",
        "Boltz-2": "Boltz",

        "Chai": "Chai",
        "Chai-1": "Chai",

        "Protenix": "Protenix",

        "IntelliFold": "IntelliFold",
        "IntelliFold2": "IntelliFold",
    }

    results["predictor"] = (
        results["predictor"]
        .replace(predictor_mapping)
    )


    # ========================================================
    # CHECK DUPLICATES
    # ========================================================

    print(
        "\nChecking duplicate "
        "predictor/structure combinations..."
    )

    duplicates = results.duplicated(
        subset=[
            "pdb_id",
            "predictor"
        ],
        keep=False
    )

    if duplicates.any():

        duplicate_rows = results.loc[
            duplicates,
            [
                "pdb_id",
                "predictor"
            ]
        ]

        print(
            "\nWARNING: duplicate combinations detected:"
        )

        print(
            duplicate_rows.to_string(
                index=False
            )
        )

        print(
            "\nKeeping the last occurrence."
        )

        results = results.drop_duplicates(
            subset=[
                "pdb_id",
                "predictor"
            ],
            keep="last"
        )

    else:

        print(
            "No duplicates found."
        )


    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    results = clean_numeric_columns(
        results
    )


    # ========================================================
    # EXECUTION TIME
    # ========================================================

    if "execution_time" in results.columns:

        results["execution_seconds"] = (
            results["execution_time"]
            .apply(parse_execution_time)
        )

    else:

        results["execution_time"] = np.nan
        results["execution_seconds"] = np.nan


    # ========================================================
    # STATUS
    # ========================================================

    if "status" not in results.columns:

        results["status"] = "FAIL"

    results["status"] = (
        results["status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    results["prediction_success"] = (
        results["status"] == "SUCCESS"
    )


    # ========================================================
    # GET STRUCTURES
    # ========================================================

    metadata_structures = set(
        metadata["pdb_id"]
        .dropna()
        .unique()
    )

    result_structures = set(
        results["pdb_id"]
        .dropna()
        .unique()
    )

    all_structures = sorted(
        metadata_structures.union(
            result_structures
        )
    )

    print("\nStructures found")
    print("-" * 70)

    print(
        f"From metadata:       "
        f"{len(metadata_structures)}"
    )

    print(
        f"From OpenStructure:  "
        f"{len(result_structures)}"
    )

    print(
        f"Total structures:    "
        f"{len(all_structures)}"
    )


    # ========================================================
    # BUILD COMPLETE PREDICTOR MATRIX
    # ========================================================

    print(
        "\nBuilding complete predictor matrix..."
    )

    complete_rows = []

    for pdb_id in all_structures:

        existing = results[
            results["pdb_id"] == pdb_id
        ]

        metadata_match = metadata[
            metadata["pdb_id"] == pdb_id
        ]

        if not metadata_match.empty:

            category = metadata_match.iloc[0][
                "category"
            ]

        elif not existing.empty:

            category = existing.iloc[0][
                "category"
            ]

        else:

            category = np.nan

        for predictor in PREDICTORS:

            predictor_row = existing[
                existing["predictor"] == predictor
            ]

            if not predictor_row.empty:

                row = predictor_row.iloc[0].copy()

                row["category"] = category

            else:

                row = {
                    "pdb_id": pdb_id,
                    "category": category,
                    "predictor": predictor,
                    "execution_time": np.nan,
                    "execution_seconds": np.nan,
                    "status": "FAIL",
                    "prediction_success": False,
                }

                for metric in METRICS:

                    row[metric] = np.nan

            complete_rows.append(row)

    benchmark = pd.DataFrame(
        complete_rows
    )


    # ========================================================
    # REORDER COLUMNS
    # ========================================================

    preferred_columns = [
        "pdb_id",
        "category",
        "predictor",
        "execution_time",
        "execution_seconds",
    ]

    other_columns = [
        column
        for column in benchmark.columns
        if column not in preferred_columns
    ]

    benchmark = benchmark[
        preferred_columns + other_columns
    ]


    # ========================================================
    # RE-CLEAN NUMERIC COLUMNS
    # ========================================================

    benchmark = clean_numeric_columns(
        benchmark
    )


    # ========================================================
    # STRUCTURE-LEVEL SUCCESS
    # ========================================================

    print(
        "\nCalculating structure-level success..."
    )

    success_table = (
        benchmark
        .pivot(
            index="pdb_id",
            columns="predictor",
            values="prediction_success"
        )
        .reindex(columns=PREDICTORS)
        .fillna(False)
    )

    success_table[
        "n_successful_predictors"
    ] = (
        success_table[PREDICTORS]
        .sum(axis=1)
    )

    success_table[
        "n_failed_predictors"
    ] = (
        len(PREDICTORS)
        - success_table[
            "n_successful_predictors"
        ]
    )

    success_table[
        "all_predictors_success"
    ] = (
        success_table[PREDICTORS]
        .all(axis=1)
    )

    success_table = (
        success_table.reset_index()
    )


    # ========================================================
    # MERGE STRUCTURE SUCCESS
    # ========================================================

    benchmark = benchmark.merge(
        success_table[
            [
                "pdb_id",
                "n_successful_predictors",
                "n_failed_predictors",
                "all_predictors_success",
            ]
        ],
        on="pdb_id",
        how="left"
    )


    # ========================================================
    # MERGE METADATA
    # ========================================================

    print(
        "\nMerging RCSB metadata..."
    )

    metadata_columns = [
        column
        for column in metadata.columns
        if column != "category"
    ]

    metadata_for_merge = (
        metadata[
            metadata_columns
        ]
        .drop_duplicates(
            subset=["pdb_id"],
            keep="first"
        )
    )

    benchmark = benchmark.merge(
        metadata_for_merge,
        on="pdb_id",
        how="left"
    )


    # ========================================================
    # FIX CATEGORY AFTER MERGE
    # ========================================================

    if (
        "category_x" in benchmark.columns
        and
        "category_y" in benchmark.columns
    ):

        benchmark["category"] = (
            benchmark["category_y"]
            .fillna(
                benchmark["category_x"]
            )
        )

        benchmark.drop(
            columns=[
                "category_x",
                "category_y"
            ],
            inplace=True
        )

    benchmark["category"] = (
        benchmark["category"]
        .apply(standardize_category)
    )


    # ========================================================
    # ADD SIZE VARIABLES
    # ========================================================

    if "structure_residues" in benchmark.columns:

        benchmark["structure_residues"] = (
            pd.to_numeric(
                benchmark["structure_residues"],
                errors="coerce"
            )
        )

        benchmark[
            "structure_residues_k"
        ] = (
            benchmark["structure_residues"]
            / 1000
        )


    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    main_columns = [
        "pdb_id",
        "category",
        "predictor",

        "status",
        "prediction_success",

        "execution_time",
        "execution_seconds",

        "n_successful_predictors",
        "n_failed_predictors",
        "all_predictors_success",
    ]

    remaining_columns = [
        column
        for column in benchmark.columns
        if column not in main_columns
    ]

    benchmark = benchmark[
        main_columns + remaining_columns
    ]


    # ========================================================
    # SORT
    # ========================================================

    category_order = [
        "Protein-Protein",
        "Protein-Antibody",
        "Protein-DNA",
        "Protein-RNA",
        "Protein-Peptide",
    ]

    benchmark["category"] = pd.Categorical(
        benchmark["category"],
        categories=category_order,
        ordered=True
    )

    benchmark["predictor"] = pd.Categorical(
        benchmark["predictor"],
        categories=PREDICTORS,
        ordered=True
    )

    benchmark = benchmark.sort_values(
        [
            "category",
            "pdb_id",
            "predictor"
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # SAVE COMPLETE DATASET
    # ========================================================

    complete_dataset_file = os.path.join(
        output_dir,
        "benchmark_dataset.csv"
    )

    benchmark.to_csv(
        complete_dataset_file,
        index=False
    )

    print(
        f"\nSaved complete dataset: "
        f"{complete_dataset_file}"
    )


    # ========================================================
    # SUMMARY BY PREDICTOR
    # ========================================================

    summary_predictor = []

    for predictor in PREDICTORS:

        subset = benchmark[
            benchmark["predictor"] == predictor
        ]

        n_structures = (
            subset["pdb_id"].nunique()
        )

        n_success = subset[
            "prediction_success"
        ].sum()

        n_fail = (
            n_structures
            - n_success
        )

        success_rate = (
            n_success
            / n_structures
            * 100
            if n_structures > 0
            else np.nan
        )

        summary_predictor.append({

            "predictor": predictor,

            "n_structures": n_structures,

            "n_success": n_success,

            "n_fail": n_fail,

            "success_rate_percent":
                success_rate,
        })

    summary_predictor = pd.DataFrame(
        summary_predictor
    )

    summary_predictor.to_csv(
        os.path.join(
            output_dir,
            "summary_by_predictor.csv"
        ),
        index=False
    )


    # ========================================================
    # SUMMARY BY CATEGORY
    # ========================================================

    category_summary = []

    for category in category_order:

        subset = benchmark[
            benchmark["category"] == category
        ]

        n_structures = (
            subset["pdb_id"].nunique()
        )

        category_summary.append({
            "category": category,
            "n_structures": n_structures,
        })

    category_summary = pd.DataFrame(
        category_summary
    )

    category_summary.to_csv(
        os.path.join(
            output_dir,
            "summary_by_category.csv"
        ),
        index=False
    )


    # ========================================================
    # SUCCESS RATES BY CATEGORY/PREDICTOR
    # ========================================================

    success_rows = []

    for category in category_order:

        for predictor in PREDICTORS:

            subset = benchmark[
                (benchmark["category"] == category)
                &
                (benchmark["predictor"] == predictor)
            ]

            n_structures = (
                subset["pdb_id"].nunique()
            )

            n_success = subset[
                "prediction_success"
            ].sum()

            success_rate = (
                n_success
                / n_structures
                * 100
                if n_structures > 0
                else np.nan
            )

            success_rows.append({

                "category": category,

                "predictor": predictor,

                "n_structures": n_structures,

                "n_success": n_success,

                "n_fail":
                    n_structures - n_success,

                "success_rate_percent":
                    success_rate,
            })

    success_rates = pd.DataFrame(
        success_rows
    )

    success_rates.to_csv(
        os.path.join(
            output_dir,
            "success_rates.csv"
        ),
        index=False
    )


    # ========================================================
    # FAILURE DETAILS BY PREDICTOR
    # ========================================================

    print(
        "\nChecking failed predictions by predictor..."
    )

    failure_rows = []

    for predictor in PREDICTORS:

        subset = benchmark[
            benchmark["predictor"] == predictor
        ].copy()

        failures = subset[
            ~subset["prediction_success"]
        ].copy()

        print(
            f"\n{predictor}: "
            f"{len(failures)} failures"
        )

        if len(failures) == 0:

            print("  None")
            continue

        print(
            failures["pdb_id"].tolist()
        )

        for _, row in failures.iterrows():

            failure_rows.append({

                "pdb_id": row["pdb_id"],

                "category": row["category"],

                "predictor": predictor,

                "status": row["status"],

                "execution_time":
                    row.get(
                        "execution_time",
                        np.nan
                    ),

                "execution_seconds":
                    row.get(
                        "execution_seconds",
                        np.nan
                    ),
            })

    failure_details = pd.DataFrame(
        failure_rows
    )

    failure_details = failure_details.sort_values(
        [
            "predictor",
            "category",
            "pdb_id",
        ]
    )

    failure_details.to_csv(
        os.path.join(
            output_dir,
            "failed_predictions.csv"
        ),
        index=False
    )


    # ========================================================
    # FAILURE LIST BY PREDICTOR
    # ========================================================

    failure_pdb_table = []

    for predictor in PREDICTORS:

        subset = failure_details[
            failure_details["predictor"] == predictor
        ]

        failure_pdb_table.append({

            "predictor": predictor,

            "n_failures": len(subset),

            "failed_pdb_ids": ";".join(
                subset["pdb_id"].astype(str)
            ),
        })

    failure_pdb_table = pd.DataFrame(
        failure_pdb_table
    )

    failure_pdb_table.to_csv(
        os.path.join(
            output_dir,
            "failed_pdbs_by_predictor.csv"
        ),
        index=False
    )


    # ========================================================
    # RUNTIME SUMMARY
    # ========================================================

    runtime_rows = []

    for predictor in PREDICTORS:

        subset = benchmark[
            (benchmark["predictor"] == predictor)
            &
            (benchmark["prediction_success"])
            &
            (benchmark["execution_seconds"].notna())
        ]

        if subset.empty:

            runtime_rows.append({

                "predictor": predictor,
                "n": 0,
                "mean_seconds": np.nan,
                "median_seconds": np.nan,
                "std_seconds": np.nan,
                "min_seconds": np.nan,
                "max_seconds": np.nan,
            })

        else:

            runtime_rows.append({

                "predictor": predictor,

                "n": len(subset),

                "mean_seconds":
                    subset[
                        "execution_seconds"
                    ].mean(),

                "median_seconds":
                    subset[
                        "execution_seconds"
                    ].median(),

                "std_seconds":
                    subset[
                        "execution_seconds"
                    ].std(),

                "min_seconds":
                    subset[
                        "execution_seconds"
                    ].min(),

                "max_seconds":
                    subset[
                        "execution_seconds"
                    ].max(),
            })

    runtime_summary = pd.DataFrame(
        runtime_rows
    )

    runtime_summary.to_csv(
        os.path.join(
            output_dir,
            "execution_time_summary.csv"
        ),
        index=False
    )


    # ========================================================
    # METRIC SUMMARY BY PREDICTOR
    # ========================================================

    metric_summary = []

    for predictor in PREDICTORS:

        subset = benchmark[
            (benchmark["predictor"] == predictor)
            &
            (benchmark["prediction_success"])
        ]

        for metric in METRICS:

            if metric not in subset.columns:
                continue

            values = subset[
                metric
            ].dropna()

            if values.empty:
                continue

            metric_summary.append({

                "predictor": predictor,
                "metric": metric,

                "n": len(values),

                "mean": values.mean(),
                "std": values.std(),

                "median": values.median(),

                "q25":
                    values.quantile(0.25),

                "q75":
                    values.quantile(0.75),

                "min": values.min(),
                "max": values.max(),
            })

    metric_summary = pd.DataFrame(
        metric_summary
    )

    metric_summary.to_csv(
        os.path.join(
            output_dir,
            "metric_summary_by_predictor.csv"
        ),
        index=False
    )


    # ========================================================
    # METRIC SUMMARY BY CATEGORY
    # ========================================================

    metric_category_summary = []

    for category in category_order:

        for predictor in PREDICTORS:

            subset = benchmark[
                (benchmark["category"] == category)
                &
                (benchmark["predictor"] == predictor)
                &
                (benchmark["prediction_success"])
            ]

            for metric in METRICS:

                if metric not in subset.columns:
                    continue

                values = subset[
                    metric
                ].dropna()

                if values.empty:
                    continue

                metric_category_summary.append({

                    "category": category,

                    "predictor": predictor,

                    "metric": metric,

                    "n": len(values),

                    "mean": values.mean(),
                    "std": values.std(),

                    "median":
                        values.median(),

                    "q25":
                        values.quantile(0.25),

                    "q75":
                        values.quantile(0.75),

                    "min": values.min(),
                    "max": values.max(),
                })

    metric_category_summary = pd.DataFrame(
        metric_category_summary
    )

    metric_category_summary.to_csv(
        os.path.join(
            output_dir,
            "metric_summary_by_category.csv"
        ),
        index=False
    )


    # ========================================================
    # STRUCTURE-LEVEL METADATA
    # ========================================================

    structure_metadata_columns = [
        "pdb_id",
        "category",
        "n_successful_predictors",
        "n_failed_predictors",
        "all_predictors_success",
    ]

    metadata_candidates = [
        "experimental_method",
        "resolution",
        "assembly_count",
        "structure_chain_count",
        "structure_chain_ids",
        "structure_chain_types",
        "structure_chain_lengths",
        "structure_residues",
        "structure_residues_k",
    ]

    for column in metadata_candidates:

        if column in benchmark.columns:

            structure_metadata_columns.append(
                column
            )

    structure_metadata = (
        benchmark[
            structure_metadata_columns
        ]
        .drop_duplicates(
            subset=["pdb_id"]
        )
        .sort_values(
            ["category", "pdb_id"]
        )
    )

    structure_metadata.to_csv(
        os.path.join(
            output_dir,
            "structure_level_metadata.csv"
        ),
        index=False
    )


    # ========================================================
    # DATASET STATISTICS TXT
    # ========================================================

    stats_file = os.path.join(
        output_dir,
        "dataset_statistics.txt"
    )

    with open(stats_file, "w") as f:

        f.write(
            "BENCHMARK DATASET STATISTICS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Total structures: "
            f"{len(all_structures)}\n"
        )

        f.write(
            f"Expected predictors per structure: "
            f"{len(PREDICTORS)}\n"
        )

        f.write(
            f"Expected predictor rows: "
            f"{len(all_structures) * len(PREDICTORS)}\n"
        )

        f.write("\n")
        f.write("STRUCTURES BY CATEGORY\n")
        f.write("-" * 70 + "\n")

        category_counts = (
            structure_metadata
            .groupby("category")
            .size()
        )

        for category, count in category_counts.items():

            f.write(
                f"{category}: {count}\n"
            )

        f.write("\n")
        f.write("SUCCESS RATES\n")
        f.write("-" * 70 + "\n")

        for _, row in summary_predictor.iterrows():

            f.write(
                f"{row['predictor']}: "
                f"{row['n_success']}/"
                f"{row['n_structures']} "
                f"({row['success_rate_percent']:.2f}%)\n"
            )

        f.write("\n")
        f.write("COMPLETE STRUCTURES\n")
        f.write("-" * 70 + "\n")

        n_complete = success_table[
            "all_predictors_success"
        ].sum()

        f.write(
            f"All five predictors successful: "
            f"{n_complete}/"
            f"{len(all_structures)} "
            f"({n_complete / len(all_structures) * 100:.2f}%)\n"
        )

        f.write("\n")
        f.write(
            "STRUCTURES WITH AT LEAST ONE FAILURE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        n_incomplete = (
            len(all_structures)
            - n_complete
        )

        f.write(
            f"{n_incomplete}/"
            f"{len(all_structures)} "
            f"({n_incomplete / len(all_structures) * 100:.2f}%)\n"
        )


    # ========================================================
    # FINAL CHECKS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL CHECKS"
    )

    print(
        "=" * 70
    )

    expected_rows = (
        len(all_structures)
        * len(PREDICTORS)
    )

    actual_rows = len(benchmark)

    print(
        f"Expected rows: {expected_rows}"
    )

    print(
        f"Actual rows:   {actual_rows}"
    )

    if expected_rows == actual_rows:

        print(
            "? Complete predictor matrix"
        )

    else:

        print(
            "? WARNING: unexpected number of rows"
        )

    print(
        "\nRows by predictor:"
    )

    print(
        benchmark["predictor"]
        .value_counts()
        .reindex(PREDICTORS)
        .to_string()
    )

    print(
        "\nSuccesses by predictor:"
    )

    print(
        benchmark
        .groupby("predictor")[
            "prediction_success"
        ]
        .sum()
        .reindex(PREDICTORS)
        .to_string()
    )

    print(
        "\nStructures with all predictors successful:"
    )

    n_complete = success_table[
        "all_predictors_success"
    ].sum()

    print(
        f"{n_complete}/"
        f"{len(all_structures)} "
        f"({n_complete / len(all_structures) * 100:.2f}%)"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAll analysis files were written to:"
    )

    print(
        f"  {output_dir}/"
    )

    print(
        "\nMain file:"
    )

    print(
        "  benchmark_dataset.csv"
    )

    print(
        "\nSummary files:"
    )

    print(
        "  summary_by_predictor.csv"
    )

    print(
        "  summary_by_category.csv"
    )

    print(
        "  success_rates.csv"
    )

    print(
        "  execution_time_summary.csv"
    )

    print(
        "  metric_summary_by_predictor.csv"
    )

    print(
        "  metric_summary_by_category.csv"
    )

    print(
        "  structure_level_metadata.csv"
    )

    print(
        "  dataset_statistics.txt"
    )


if __name__ == "__main__":
    main()