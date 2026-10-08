# -*- coding: utf-8 -*-

"""
Extreme-case analysis for the BioFoldBenchmark.

This script:
1. Identifies structures with large disagreement among predictors based on
   lDDT, TM-score, RMSD and DockQ.
2. Selects extreme cases while ensuring representation of all complex classes.
3. Locates ChimeraProtDiscrepancies outputs in Scipion projects.
4. Parses per-residue C-alpha deviations.
5. Preserves raw Chimera observations, including model-only residues.
6. Excludes residues without any reference correspondence only for
   residue-level downstream analyses.
7. Summarizes residue-level deviations.
8. Compares predictors in the selected extreme cases.
9. Identifies localized regions of elevated deviation.
10. Generates plots and summary tables.

Important:
- Global structure-disagreement selection DOES include model-only terminal
  extensions, because the selection is based on structure-level metrics.
- Model-only residues are excluded only from residue-level analyses when no
  predictor has a valid reference correspondence for that residue.
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

PREDICTORS = [
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold",
    "AlphaFold3",
]

CATEGORIES = [
    "Protein-Protein",
    "Protein-Antibody",
    "Protein-DNA",
    "Protein-RNA",
    "Protein-Peptide",
]

PRIMARY_METRICS = [
    "lddt",
    "tm_score",
    "rmsd",
    "dockq_ave",
]

N_EXTREME_CASES = 10

# Minimum number of predictors with at least one available primary metric.
MIN_SUCCESSFUL_PREDICTORS = 3

# Residue-level analysis parameters.
WINDOW_SIZE = 7
MIN_REGION_LENGTH = 4
HIGH_DEVIATION_PERCENTILE = 0.90


# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

def ensure_output_directories(output_dir):
    """Create output directories if they do not already exist."""

    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(
        os.path.join(output_dir, "plots"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(output_dir, "residue_plots"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(output_dir, "heatmaps"),
        exist_ok=True,
    )


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def clean_pdb_id(value):
    """Normalize a PDB identifier."""

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    # Remove common filename/project decorations if present.
    value = value.replace(".PDB", "")

    return value


def safe_numeric(value):
    """Convert a value to float, returning NaN when conversion fails."""

    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {
            "",
            "nan",
            "none",
            "null",
            "na",
            "n/a",
        }:
            return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================================
# LOAD BENCHMARK DATASET
# ============================================================================

def load_benchmark_dataset(dataset_file):
    """
    Load the benchmark dataset containing predictor-level accuracy metrics.
    """

    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(
            f"Benchmark dataset not found:\n{dataset_file}"
        )

    df = pd.read_csv(dataset_file)

    # Normalize column names.
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Try to identify PDB column.
    pdb_candidates = [
        "pdb_id",
        "PDB",
        "pdb",
        "PDB_ID",
        "pdbid",
    ]

    pdb_column = None

    for column in pdb_candidates:
        if column in df.columns:
            pdb_column = column
            break

    if pdb_column is None:
        raise ValueError(
            "Could not identify the PDB identifier column in "
            f"{dataset_file}. Available columns:\n{list(df.columns)}"
        )

    if pdb_column != "pdb_id":
        df = df.rename(
            columns={pdb_column: "pdb_id"}
        )

    df["pdb_id"] = df["pdb_id"].apply(clean_pdb_id)

    # Try to identify predictor column.
    predictor_candidates = [
        "predictor",
        "Predictor",
        "method",
        "Method",
    ]

    predictor_column = None

    for column in predictor_candidates:
        if column in df.columns:
            predictor_column = column
            break

    if predictor_column is None:
        raise ValueError(
            "Could not identify the predictor column in "
            f"{dataset_file}. Available columns:\n{list(df.columns)}"
        )

    if predictor_column != "predictor":
        df = df.rename(
            columns={predictor_column: "predictor"}
        )

    df["predictor"] = (
        df["predictor"]
        .astype(str)
        .str.strip()
    )

    # Normalize metric columns.
    for metric in PRIMARY_METRICS:
        if metric in df.columns:
            df[metric] = df[metric].apply(safe_numeric)
        else:
            warnings.warn(
                f"Metric '{metric}' not found in benchmark dataset."
            )
            df[metric] = np.nan

    # Try to identify category column.
    category_candidates = [
        "category",
        "Category",
        "complex_category",
        "complex_type",
        "Complex",
    ]

    category_column = None

    for column in category_candidates:
        if column in df.columns:
            category_column = column
            break

    if category_column is not None and category_column != "category":
        df = df.rename(
            columns={category_column: "category"}
        )

    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================================
# STRUCTURE-LEVEL DISAGREEMENT
# ============================================================================

def calculate_structure_disagreement(df):
    """
    Calculate a structure-level disagreement score.

    For each PDB:
    - Determine the range between the highest and lowest predictor value
      for each primary metric.
    - Standardize each metric-specific range across structures using z-scores.
    - Average the absolute z-scores across available metrics.

    Important:
    This analysis uses structure-level benchmark metrics and therefore does
    NOT filter out model-only terminal extensions.
    """

    records = []

    for pdb_id, group in df.groupby("pdb_id"):

        metric_ranges = {}

        available_predictors = []

        for predictor in PREDICTORS:

            predictor_rows = group[
                group["predictor"].str.lower()
                == predictor.lower()
            ]

            if predictor_rows.empty:
                continue

            row = predictor_rows.iloc[0]

            has_metric = False

            for metric in PRIMARY_METRICS:

                value = safe_numeric(
                    row.get(metric, np.nan)
                )

                if not np.isnan(value):
                    has_metric = True
                    break

            if has_metric:
                available_predictors.append(
                    predictor
                )

        if len(available_predictors) < MIN_SUCCESSFUL_PREDICTORS:
            continue

        for metric in PRIMARY_METRICS:

            values = []

            for predictor in available_predictors:

                predictor_rows = group[
                    group["predictor"].str.lower()
                    == predictor.lower()
                ]

                if predictor_rows.empty:
                    continue

                value = safe_numeric(
                    predictor_rows.iloc[0].get(
                        metric,
                        np.nan,
                    )
                )

                if not np.isnan(value):
                    values.append(value)

            if len(values) < 2:
                continue

            metric_ranges[metric] = (
                max(values) - min(values)
            )

        # Require at least two metrics.
        if len(metric_ranges) < 2:
            continue

        record = {
            "pdb_id": pdb_id,
            "n_predictors": len(available_predictors),
        }

        for metric in PRIMARY_METRICS:
            record[f"{metric}_range"] = metric_ranges.get(
                metric,
                np.nan,
            )

        records.append(record)

    disagreement_df = pd.DataFrame(records)

    if disagreement_df.empty:
        raise RuntimeError(
            "No structures contained enough successful predictors "
            "and primary metrics for disagreement analysis."
        )

    # ------------------------------------------------------------------------
    # Standardize metric-specific ranges across structures.
    # ------------------------------------------------------------------------

    z_columns = []

    for metric in PRIMARY_METRICS:

        range_column = f"{metric}_range"

        if range_column not in disagreement_df.columns:
            continue

        values = disagreement_df[range_column]

        valid = values.notna()

        if valid.sum() < 2:
            continue

        mean_value = values[valid].mean()
        std_value = values[valid].std(ddof=0)

        z_column = f"{metric}_range_z"

        if std_value == 0 or np.isnan(std_value):
            disagreement_df[z_column] = np.nan
        else:
            disagreement_df[z_column] = (
                (values - mean_value)
                / std_value
            )

        z_columns.append(z_column)

    # Mean absolute z-score.
    disagreement_df["disagreement"] = (
        disagreement_df[z_columns]
        .abs()
        .mean(axis=1)
    )

    disagreement_df = disagreement_df.sort_values(
        "disagreement",
        ascending=False,
    ).reset_index(drop=True)

    return disagreement_df


# ============================================================================
# EXTREME CASE SELECTION
# ============================================================================

def select_extreme_cases(
    disagreement_df,
    benchmark_df,
):
    """
    Select extreme structures.

    Selection strategy:
    1. Select the strongest disagreement case from each category.
    2. Fill remaining positions globally by disagreement score.

    This function intentionally does NOT apply residue-level correspondence
    filtering.
    """

    # Add category information.
    category_map = (
        benchmark_df[
            ["pdb_id", "category"]
        ]
        .drop_duplicates("pdb_id")
        .set_index("pdb_id")["category"]
        .to_dict()
    )

    selected = []

    working = disagreement_df.copy()

    working["category"] = (
        working["pdb_id"]
        .map(category_map)
    )

    # ------------------------------------------------------------------------
    # One representative from each category.
    # ------------------------------------------------------------------------

    for category in CATEGORIES:

        category_df = working[
            working["category"] == category
        ]

        if category_df.empty:
            continue

        selected.append(
            category_df.iloc[0]
        )

    selected_pdbs = {
        row["pdb_id"]
        for row in selected
    }

    # ------------------------------------------------------------------------
    # Fill remaining slots globally.
    # ------------------------------------------------------------------------

    for _, row in working.iterrows():

        if len(selected) >= N_EXTREME_CASES:
            break

        if row["pdb_id"] in selected_pdbs:
            continue

        selected.append(row)

        selected_pdbs.add(
            row["pdb_id"]
        )

    selected_df = pd.DataFrame(selected)

    if selected_df.empty:
        raise RuntimeError(
            "No extreme cases could be selected."
        )

    selected_df = selected_df.sort_values(
        "disagreement",
        ascending=False,
    ).reset_index(drop=True)

    return selected_df


# ============================================================================
# LOCATE CHIMERA OUTPUT
# ============================================================================

def find_chimera_extra(
    pdb_id,
    scipion_projects_dir,
):
    """
    Find the ChimeraProtDiscrepancies 'extra' directory for a PDB.

    Scipion projects are expected directly under the user-provided
    Scipion projects directory.

    A project is identified as:

        BioFoldBenchmark_*_<PDB>

    Chimera outputs may be nested below:

        Runs/<run>_ChimeraProtDiscrepancies/extra/

    and RMSD files may be nested further inside predictor/model folders.
    """

    pdb_id = clean_pdb_id(pdb_id)

    if not os.path.isdir(scipion_projects_dir):
        warnings.warn(
            "Scipion projects directory does not exist:\n"
            f"{scipion_projects_dir}"
        )
        return None

    candidate_projects = []

    try:
        project_names = os.listdir(
            scipion_projects_dir
        )
    except OSError as exc:
        warnings.warn(
            f"Could not list Scipion projects directory: {exc}"
        )
        return None

    for project_name in project_names:

        project_path = os.path.join(
            scipion_projects_dir,
            project_name,
        )

        if not os.path.isdir(project_path):
            continue

        if not project_name.startswith(
            "BioFoldBenchmark_"
        ):
            continue

        project_pdb = clean_pdb_id(
            project_name.rsplit("_", 1)[-1]
        )

        if project_pdb != pdb_id:
            continue

        candidate_projects.append(
            project_path
        )

    if not candidate_projects:
        warnings.warn(
            f"No BioFoldBenchmark Scipion project found for {pdb_id}"
        )
        return None

    # ------------------------------------------------------------------------
    # Find ChimeraProtDiscrepancies extra directories.
    # ------------------------------------------------------------------------

    extra_candidates = []

    for project_path in candidate_projects:

        runs_dir = os.path.join(
            project_path,
            "Runs",
        )

        if not os.path.isdir(runs_dir):
            continue

        try:
            run_names = os.listdir(runs_dir)
        except OSError:
            continue

        for run_name in run_names:

            if "ChimeraProtDiscrepancies" not in run_name:
                continue

            run_path = os.path.join(
                runs_dir,
                run_name,
            )

            if not os.path.isdir(run_path):
                continue

            extra_dir = os.path.join(
                run_path,
                "extra",
            )

            if not os.path.isdir(extra_dir):
                continue

            # Count recognizable RMSD files recursively.
            recognized_files = []

            for root, _, files in os.walk(
                extra_dir
            ):
                for filename in files:

                    if not filename.endswith(".txt"):
                        continue

                    match = re.match(
                        r"rmsd_model_00_model_\d+_(.+?)_chain_(.+?)\.txt$",
                        filename,
                    )

                    if match:
                        recognized_files.append(
                            os.path.join(
                                root,
                                filename,
                            )
                        )

            if recognized_files:
                extra_candidates.append(
                    (
                        len(recognized_files),
                        extra_dir,
                    )
                )

    if not extra_candidates:
        warnings.warn(
            f"No usable ChimeraProtDiscrepancies output found for {pdb_id}"
        )
        return None

    # Prefer the run with the largest number of recognizable RMSD files.
    extra_candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return extra_candidates[0][1]


# ============================================================================
# PARSE RMSD FILE
# ============================================================================

def parse_rmsd_file(
    filename,
    pdb_id,
    predictor,
    chain,
):
    """
    Parse a Chimera RMSD file.

    Expected format:

        residue: value

    or:

        residue: None

    'None' is converted to NaN.

    NaN values are intentionally retained because they identify residues
    without a valid reference correspondence for that predictor.
    """

    records = []

    if not os.path.isfile(filename):
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "predictor",
                "chain",
                "residue",
                "deviation",
            ]
        )

    try:
        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:
                    continue

                if ":" not in line:
                    continue

                residue_text, value_text = (
                    line.split(":", 1)
                )

                residue_text = residue_text.strip()
                value_text = value_text.strip()

                # Extract integer residue number.
                residue_match = re.search(
                    r"-?\d+",
                    residue_text,
                )

                if residue_match is None:
                    continue

                residue = int(
                    residue_match.group()
                )

                if value_text.lower() in {
                    "none",
                    "nan",
                    "null",
                    "na",
                }:
                    deviation = np.nan
                else:
                    deviation = safe_numeric(
                        value_text
                    )

                records.append(
                    {
                        "pdb_id": pdb_id,
                        "predictor": predictor,
                        "chain": chain,
                        "residue": residue,
                        "deviation": deviation,
                    }
                )

    except OSError as exc:
        warnings.warn(
            f"Could not read RMSD file {filename}: {exc}"
        )

    return pd.DataFrame(records)


# ============================================================================
# LOAD CHIMERA RESIDUE DATA
# ============================================================================

def load_chimera_residue_data(
    selected_cases,
    scipion_projects_dir,
):
    """
    Load residue-level Chimera deviations for all selected cases.

    The selected 'extra' directory is searched recursively because Scipion
    stores RMSD files inside model/predictor subdirectories.
    """

    all_records = []

    for _, case in selected_cases.iterrows():

        pdb_id = clean_pdb_id(
            case["pdb_id"]
        )

        extra_dir = find_chimera_extra(
            pdb_id,
            scipion_projects_dir,
        )

        if extra_dir is None:
            continue

        recognized_files = []

        for root, _, files in os.walk(
            extra_dir
        ):
            for filename in files:

                if not filename.endswith(".txt"):
                    continue

                match = re.match(
                    r"rmsd_model_00_model_\d+_(.+?)_chain_(.+?)\.txt$",
                    filename,
                )

                if match is None:
                    continue

                predictor = match.group(1)
                chain = match.group(2)

                recognized_files.append(
                    (
                        os.path.join(
                            root,
                            filename,
                        ),
                        predictor,
                        chain,
                    )
                )

        if not recognized_files:
            warnings.warn(
                f"No RMSD files found recursively for {pdb_id}"
            )
            continue

        for (
            filename,
            predictor,
            chain,
        ) in recognized_files:

            # Normalize predictor names.
            predictor_normalized = None

            for expected_predictor in PREDICTORS:

                if (
                    predictor.lower()
                    == expected_predictor.lower()
                ):
                    predictor_normalized = (
                        expected_predictor
                    )
                    break

            if predictor_normalized is None:

                # Handle possible naming variants.
                predictor_lower = predictor.lower()

                if "alpha" in predictor_lower:
                    predictor_normalized = (
                        "AlphaFold3"
                    )
                elif "boltz" in predictor_lower:
                    predictor_normalized = (
                        "Boltz"
                    )
                elif "chai" in predictor_lower:
                    predictor_normalized = (
                        "Chai"
                    )
                elif "protenix" in predictor_lower:
                    predictor_normalized = (
                        "Protenix"
                    )
                elif "intellifold" in predictor_lower:
                    predictor_normalized = (
                        "IntelliFold"
                    )
                else:
                    predictor_normalized = predictor

            parsed = parse_rmsd_file(
                filename=filename,
                pdb_id=pdb_id,
                predictor=predictor_normalized,
                chain=chain,
            )

            if not parsed.empty:
                all_records.append(
                    parsed
                )

    if not all_records:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "predictor",
                "chain",
                "residue",
                "deviation",
            ]
        )

    residue_df = pd.concat(
        all_records,
        ignore_index=True,
    )

    residue_df = residue_df.drop_duplicates(
        subset=[
            "pdb_id",
            "predictor",
            "chain",
            "residue",
        ],
        keep="first",
    )

    residue_df = residue_df.sort_values(
        [
            "pdb_id",
            "chain",
            "residue",
            "predictor",
        ]
    ).reset_index(drop=True)

    return residue_df


# ============================================================================
# FILTER RESIDUES WITHOUT REFERENCE CORRESPONDENCE
# ============================================================================

def filter_reference_corresponding_residues(
    residue_df,
):
    """
    Remove residue positions for which ALL predictors have NaN deviation.

    Interpretation:
    - If at least one predictor has a numeric deviation, the residue position
      is retained.
    - If every predictor has NaN at that residue, the position is considered
      to lack reference correspondence and is excluded from downstream
      residue-level comparisons.

    This function is deliberately NOT applied to the raw residue dataframe.
    """

    if residue_df.empty:
        return residue_df.copy()

    required_columns = {
        "pdb_id",
        "chain",
        "residue",
        "deviation",
    }

    missing = required_columns.difference(
        residue_df.columns
    )

    if missing:
        raise ValueError(
            "Residue dataframe is missing columns: "
            f"{sorted(missing)}"
        )

    valid_residues = (
        residue_df
        .groupby(
            [
                "pdb_id",
                "chain",
                "residue",
            ]
        )["deviation"]
        .apply(
            lambda x: x.notna().any()
        )
        .reset_index(
            name="has_reference_correspondence"
        )
    )

    valid_residues = valid_residues[
        valid_residues[
            "has_reference_correspondence"
        ]
    ].drop(
        columns=[
            "has_reference_correspondence"
        ]
    )

    filtered = residue_df.merge(
        valid_residues,
        on=[
            "pdb_id",
            "chain",
            "residue",
        ],
        how="inner",
    )

    return filtered


# ============================================================================
# RESIDUE-LEVEL SUMMARY
# ============================================================================

def summarize_residue_deviations(
    residue_df,
):
    """
    Summarize residue-level deviations.

    Only reference-corresponding residue positions are included.
    """

    if residue_df.empty:
        return pd.DataFrame()

    df = filter_reference_corresponding_residues(
        residue_df
    )

    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(
            [
                "pdb_id",
                "chain",
                "residue",
            ]
        )
        .agg(
            mean_deviation=(
                "deviation",
                "mean",
            ),
            median_deviation=(
                "deviation",
                "median",
            ),
            max_deviation=(
                "deviation",
                "max",
            ),
            n_predictors_with_value=(
                "deviation",
                lambda x: x.notna().sum(),
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "pdb_id",
            "chain",
            "residue",
        ]
    ).reset_index(drop=True)


# ============================================================================
# PREDICTOR COMPARISON IN EXTREME CASES
# ============================================================================

def compare_predictors_extreme_cases(
    residue_df,
):
    """
    Compare residue-level deviations between predictors in the selected
    extreme cases.

    Model-only residues are excluded if no predictor has a valid reference
    correspondence.
    """

    if residue_df.empty:
        return pd.DataFrame()

    df = filter_reference_corresponding_residues(
        residue_df
    )

    if df.empty:
        return pd.DataFrame()

    records = []

    for (
        pdb_id,
        chain,
    ), group in df.groupby(
        [
            "pdb_id",
            "chain",
        ]
    ):

        for predictor in PREDICTORS:

            predictor_values = group.loc[
                group["predictor"]
                .str.lower()
                == predictor.lower(),
                "deviation",
            ]

            predictor_values = (
                predictor_values.dropna()
            )

            if predictor_values.empty:
                continue

            records.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "predictor": predictor,
                    "n_residues": len(
                        predictor_values
                    ),
                    "mean_deviation": (
                        predictor_values.mean()
                    ),
                    "median_deviation": (
                        predictor_values.median()
                    ),
                    "max_deviation": (
                        predictor_values.max()
                    ),
                }
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    return result.sort_values(
        [
            "pdb_id",
            "chain",
            "predictor",
        ]
    ).reset_index(drop=True)


# ============================================================================
# EXTREME-CASE PREDICTOR PERFORMANCE SUMMARY
# ============================================================================

def summarize_predictor_performance_extreme_cases(
    residue_df,
):
    """
    Generate a predictor-level summary over all selected extreme cases.

    Only reference-corresponding residues are included.
    """

    if residue_df.empty:
        return pd.DataFrame()

    df = filter_reference_corresponding_residues(
        residue_df
    )

    if df.empty:
        return pd.DataFrame()

    result = (
        df.groupby("predictor")
        .agg(
            n_residue_observations=(
                "deviation",
                "count",
            ),
            mean_deviation=(
                "deviation",
                "mean",
            ),
            median_deviation=(
                "deviation",
                "median",
            ),
            max_deviation=(
                "deviation",
                "max",
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "predictor"
    ).reset_index(drop=True)


# ============================================================================
# HIGH-DEVIATION REGION IDENTIFICATION
# ============================================================================

def identify_high_deviation_regions(
    residue_summary,
):
    """
    Identify localized regions of elevated residue-level deviation.

    Procedure:
    - For each PDB/chain, calculate a rolling mean using WINDOW_SIZE.
    - Define elevated deviation as values at or above the
      HIGH_DEVIATION_PERCENTILE percentile.
    - Require consecutive residue numbers.
    - Require at least MIN_REGION_LENGTH residues.
    """

    if residue_summary.empty:
        return pd.DataFrame()

    records = []

    for (
        pdb_id,
        chain,
    ), group in residue_summary.groupby(
        [
            "pdb_id",
            "chain",
        ]
    ):

        group = group.sort_values(
            "residue"
        ).copy()

        if group.empty:
            continue

        group["rolling_mean"] = (
            group["mean_deviation"]
            .rolling(
                window=WINDOW_SIZE,
                center=True,
                min_periods=1,
            )
            .mean()
        )

        valid_values = (
            group["rolling_mean"]
            .dropna()
        )

        if valid_values.empty:
            continue

        threshold = np.nanpercentile(
            valid_values,
            HIGH_DEVIATION_PERCENTILE * 100,
        )

        group["high_deviation"] = (
            group["rolling_mean"]
            >= threshold
        )

        current_region = []

        previous_residue = None

        for _, row in group.iterrows():

            residue = int(
                row["residue"]
            )

            is_high = bool(
                row["high_deviation"]
            )

            is_consecutive = (
                previous_residue is not None
                and residue
                == previous_residue + 1
            )

            if (
                is_high
                and (
                    not current_region
                    or is_consecutive
                )
            ):
                current_region.append(
                    row
                )

            else:

                if (
                    len(current_region)
                    >= MIN_REGION_LENGTH
                ):

                    region_df = pd.DataFrame(
                        current_region
                    )

                    records.append(
                        {
                            "pdb_id": pdb_id,
                            "chain": chain,
                            "start_residue": int(
                                region_df[
                                    "residue"
                                ].min()
                            ),
                            "end_residue": int(
                                region_df[
                                    "residue"
                                ].max()
                            ),
                            "region_length": len(
                                region_df
                            ),
                            "mean_deviation": (
                                region_df[
                                    "mean_deviation"
                                ].mean()
                            ),
                            "max_deviation": (
                                region_df[
                                    "mean_deviation"
                                ].max()
                            ),
                            "rolling_mean_max": (
                                region_df[
                                    "rolling_mean"
                                ].max()
                            ),
                        }
                    )

                current_region = []

                if is_high:
                    current_region.append(
                        row
                    )

            previous_residue = residue

        # Handle final region.
        if (
            len(current_region)
            >= MIN_REGION_LENGTH
        ):

            region_df = pd.DataFrame(
                current_region
            )

            records.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "start_residue": int(
                        region_df[
                            "residue"
                        ].min()
                    ),
                    "end_residue": int(
                        region_df[
                            "residue"
                        ].max()
                    ),
                    "region_length": len(
                        region_df
                    ),
                    "mean_deviation": (
                        region_df[
                            "mean_deviation"
                        ].mean()
                    ),
                    "max_deviation": (
                        region_df[
                            "mean_deviation"
                        ].max()
                    ),
                    "rolling_mean_max": (
                        region_df[
                            "rolling_mean"
                        ].max()
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "chain",
                "start_residue",
                "end_residue",
                "region_length",
                "mean_deviation",
                "max_deviation",
                "rolling_mean_max",
            ]
        )

    regions = pd.DataFrame(records)

    return regions.sort_values(
        [
            "pdb_id",
            "chain",
            "mean_deviation",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================================
# STRUCTURE DISAGREEMENT PLOT
# ============================================================================

def plot_structure_disagreement(
    selected_cases,
    output_dir,
):
    """Plot disagreement scores for selected extreme cases."""

    if selected_cases.empty:
        return

    plot_df = selected_cases.copy()

    plot_df = plot_df.sort_values(
        "disagreement",
        ascending=True,
    )

    labels = [
        f"{row.pdb_id} ({row.category})"
        for _, row in plot_df.iterrows()
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        labels,
        plot_df["disagreement"],
    )

    plt.xlabel(
        "Overall disagreement score"
    )

    plt.ylabel(
        "Structure"
    )

    plt.title(
        "Selected extreme cases"
    )

    plt.tight_layout()

    output_file = os.path.join(
        output_dir,
        "plots",
        "structure_disagreement.png",
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# RESIDUE DEVIATION PLOT
# ============================================================================

def plot_residue_deviation(
    residue_df,
    pdb_id,
    chain,
    output_dir,
):
    """
    Plot mean residue-level C-alpha deviation.

    Only reference-corresponding residue positions are plotted.
    """

    if residue_df.empty:
        return

    df = filter_reference_corresponding_residues(
        residue_df
    )

    df = df[
        (
            df["pdb_id"].str.upper()
            == pdb_id.upper()
        )
        & (
            df["chain"].astype(str)
            == str(chain)
        )
    ].copy()

    if df.empty:
        return

    summary = (
        df.groupby("residue")["deviation"]
        .mean()
        .reset_index()
    )

    if summary.empty:
        return

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        summary["residue"],
        summary["deviation"],
        linewidth=1.5,
    )

    plt.xlabel(
        "Residue"
    )

    plt.ylabel(
        "C? positional deviation (Å)"
    )

    plt.title(
        f"{pdb_id} ? chain {chain}: "
        "mean residue-level deviation"
    )

    plt.tight_layout()

    output_file = os.path.join(
        output_dir,
        "residue_plots",
        f"{pdb_id}_{chain}_residue_deviation.png",
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# PREDICTOR HEATMAP
# ============================================================================

def plot_predictor_heatmap(
    residue_df,
    pdb_id,
    chain,
    output_dir,
):
    """
    Plot predictor-specific C-alpha deviation as a residue-level heatmap.

    Only reference-corresponding residue positions are included.
    """

    if residue_df.empty:
        return

    df = filter_reference_corresponding_residues(
        residue_df
    )

    df = df[
        (
            df["pdb_id"].str.upper()
            == pdb_id.upper()
        )
        & (
            df["chain"].astype(str)
            == str(chain)
        )
    ].copy()

    if df.empty:
        return

    pivot = df.pivot_table(
        index="predictor",
        columns="residue",
        values="deviation",
        aggfunc="mean",
    )

    # Reorder predictors where available.
    available_predictors = [
        predictor
        for predictor in PREDICTORS
        if predictor in pivot.index
    ]

    if available_predictors:
        pivot = pivot.reindex(
            available_predictors
        )

    if pivot.empty:
        return

    plt.figure(
        figsize=(
            max(
                10,
                pivot.shape[1] * 0.12,
            ),
            4.5,
        )
    )

    plt.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.yticks(
        np.arange(
            len(pivot.index)
        ),
        pivot.index,
    )

    # Keep x-axis readable for long proteins.
    n_residues = len(
        pivot.columns
    )

    if n_residues <= 30:
        tick_positions = np.arange(
            n_residues
        )
    else:
        n_ticks = min(
            12,
            n_residues,
        )

        tick_positions = np.linspace(
            0,
            n_residues - 1,
            n_ticks,
            dtype=int,
        )

    plt.xticks(
        tick_positions,
        [
            str(
                pivot.columns[position]
            )
            for position in tick_positions
        ],
        rotation=45,
        ha="right",
    )

    plt.xlabel(
        "Residue"
    )

    plt.ylabel(
        "Predictor"
    )

    plt.title(
        f"{pdb_id} ? chain {chain}: "
        "C? positional deviation (Å)"
    )

    colorbar = plt.colorbar()

    colorbar.set_label(
        "C? positional deviation (Å)"
    )

    plt.tight_layout()

    output_file = os.path.join(
        output_dir,
        "heatmaps",
        f"{pdb_id}_{chain}_predictor_heatmap.png",
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# WRITE SUMMARY
# ============================================================================

def write_summary(
    selected_cases,
    residue_df,
    residue_summary,
    predictor_comparison,
    predictor_summary,
    high_deviation_regions,
    output_dir,
):
    """
    Write CSV outputs and a human-readable summary file.
    """

    # ------------------------------------------------------------------------
    # Selected cases.
    # ------------------------------------------------------------------------

    selected_cases.to_csv(
        os.path.join(
            output_dir,
            "selected_extreme_cases.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------------
    # Raw residue-level data.
    #
    # IMPORTANT:
    # This intentionally retains NaNs/model-only positions.
    # ------------------------------------------------------------------------

    residue_df.to_csv(
        os.path.join(
            output_dir,
            "residue_deviations_long.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------------
    # Filtered residue summary.
    # ------------------------------------------------------------------------

    residue_summary.to_csv(
        os.path.join(
            output_dir,
            "residue_deviation_summary.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------------
    # Predictor comparison.
    # ------------------------------------------------------------------------

    predictor_comparison.to_csv(
        os.path.join(
            output_dir,
            "extreme_case_predictor_comparison.csv",
        ),
        index=False,
    )

    predictor_summary.to_csv(
        os.path.join(
            output_dir,
            "extreme_case_predictor_summary.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------------
    # High-deviation regions.
    # ------------------------------------------------------------------------

    high_deviation_regions.to_csv(
        os.path.join(
            output_dir,
            "high_deviation_regions.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------------
    # Text summary.
    # ------------------------------------------------------------------------

    summary_file = os.path.join(
        output_dir,
        "analysis_summary.txt",
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "BioFoldBenchmark extreme-case analysis\n"
        )

        handle.write(
            "========================================\n\n"
        )

        handle.write(
            f"Number of selected cases: "
            f"{len(selected_cases)}\n\n"
        )

        handle.write(
            "Selected cases:\n"
        )

        for _, row in selected_cases.iterrows():

            handle.write(
                f"  {row['pdb_id']} | "
                f"{row.get('category', 'NA')} | "
                f"predictors={row['n_predictors']} | "
                f"disagreement="
                f"{row['disagreement']:.3f}\n"
            )

        handle.write(
            "\n"
        )

        handle.write(
            "Residue-level analysis:\n"
        )

        handle.write(
            "  Model-only residue positions were retained "
            "in the raw Chimera output.\n"
        )

        handle.write(
            "  Residue positions for which all predictors "
            "lacked a valid reference correspondence "
            "were excluded from downstream residue-level "
            "summaries, comparisons and plots.\n"
        )

        handle.write(
            "\n"
        )

        handle.write(
            "High-deviation regions:\n"
        )

        if high_deviation_regions.empty:
            handle.write(
                "  No regions met the specified criteria.\n"
            )
        else:

            for _, row in (
                high_deviation_regions.iterrows()
            ):

                handle.write(
                    f"  {row['pdb_id']} | "
                    f"chain {row['chain']} | "
                    f"{int(row['start_residue'])}-"
                    f"{int(row['end_residue'])} | "
                    f"length={int(row['region_length'])} | "
                    f"mean deviation="
                    f"{row['mean_deviation']:.3f} Å\n"
                )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extreme-case analysis for the BioFoldBenchmark."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Input benchmark_dataset.csv file."
        ),
    )

    parser.add_argument(
        "-s",
        "--scipion-projects-dir",
        required=True,
        help=(
            "Root directory containing the Scipion projects."
        ),
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=(
            "Output directory for extreme-case analysis results."
        ),
    )

    args = parser.parse_args()

    dataset_file = args.input
    scipion_projects_dir = args.scipion_projects_dir
    output_dir = args.output_dir

    print(
        "\n"
        "============================================================\n"
        "BioFoldBenchmark extreme-case analysis\n"
        "============================================================\n"
    )

    ensure_output_directories(
        output_dir
    )

    # ------------------------------------------------------------------------
    # 1. Load benchmark dataset.
    # ------------------------------------------------------------------------

    print(
        "[1/9] Loading benchmark dataset..."
    )

    benchmark_df = load_benchmark_dataset(
        dataset_file
    )

    print(
        f"      Loaded {len(benchmark_df)} predictor-level records."
    )

    # ------------------------------------------------------------------------
    # 2. Calculate structure-level disagreement.
    # ------------------------------------------------------------------------

    print(
        "[2/9] Calculating structure-level disagreement..."
    )

    disagreement_df = calculate_structure_disagreement(
        benchmark_df
    )

    disagreement_df.to_csv(
        os.path.join(
            output_dir,
            "all_structure_disagreement.csv",
        ),
        index=False,
    )

    print(
        f"      {len(disagreement_df)} structures "
        "available for disagreement analysis."
    )

    # ------------------------------------------------------------------------
    # 3. Select extreme cases.
    # ------------------------------------------------------------------------

    print(
        "[3/9] Selecting extreme cases..."
    )

    selected_cases = select_extreme_cases(
        disagreement_df,
        benchmark_df,
    )

    print(
        "\nSelected cases:"
    )

    for _, row in selected_cases.iterrows():

        print(
            f"      {row['pdb_id']} | "
            f"{row.get('category', 'NA')} | "
            f"predictors={row['n_predictors']} | "
            f"disagreement="
            f"{row['disagreement']:.3f}"
        )

    # ------------------------------------------------------------------------
    # 4. Plot structure disagreement.
    # ------------------------------------------------------------------------

    print(
        "[4/9] Plotting structure-level disagreement..."
    )

    plot_structure_disagreement(
        selected_cases,
        output_dir,
    )

    # ------------------------------------------------------------------------
    # 5. Locate and load Chimera data.
    # ------------------------------------------------------------------------

    print(
        "[5/9] Loading Chimera residue-level data..."
    )

    residue_df = load_chimera_residue_data(
        selected_cases,
        scipion_projects_dir,
    )

    print(
        f"      Loaded {len(residue_df)} residue-level observations."
    )

    if residue_df.empty:
        warnings.warn(
            "No Chimera residue-level data were found. "
            "Residue-level analyses will be empty."
        )

    # ------------------------------------------------------------------------
    # 6. Residue-level summaries.
    # ------------------------------------------------------------------------

    print(
        "[6/9] Calculating residue-level summaries..."
    )

    residue_summary = summarize_residue_deviations(
        residue_df
    )

    predictor_comparison = (
        compare_predictors_extreme_cases(
            residue_df
        )
    )

    predictor_summary = (
        summarize_predictor_performance_extreme_cases(
            residue_df
        )
    )

    # ------------------------------------------------------------------------
    # 7. High-deviation regions.
    # ------------------------------------------------------------------------

    print(
        "[7/9] Identifying high-deviation regions..."
    )

    high_deviation_regions = (
        identify_high_deviation_regions(
            residue_summary
        )
    )

    print(
        f"      Identified "
        f"{len(high_deviation_regions)} "
        "candidate high-deviation regions."
    )

    # ------------------------------------------------------------------------
    # 8. Generate residue-level plots.
    # ------------------------------------------------------------------------

    print(
        "[8/9] Generating residue-level plots..."
    )

    if not residue_df.empty:

        unique_case_chains = (
            residue_df[
                [
                    "pdb_id",
                    "chain",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "pdb_id",
                    "chain",
                ]
            )
        )

        for _, row in unique_case_chains.iterrows():

            pdb_id = row["pdb_id"]
            chain = row["chain"]

            plot_residue_deviation(
                residue_df,
                pdb_id,
                chain,
                output_dir,
            )

            plot_predictor_heatmap(
                residue_df,
                pdb_id,
                chain,
                output_dir,
            )

    # ------------------------------------------------------------------------
    # 9. Write all outputs.
    # ------------------------------------------------------------------------

    print(
        "[9/9] Writing output tables and summary..."
    )

    write_summary(
        selected_cases=selected_cases,
        residue_df=residue_df,
        residue_summary=residue_summary,
        predictor_comparison=predictor_comparison,
        predictor_summary=predictor_summary,
        high_deviation_regions=high_deviation_regions,
        output_dir=output_dir,
    )

    print(
        "\n"
        "============================================================\n"
        "Analysis completed successfully.\n"
        "============================================================\n"
        f"Output directory:\n{output_dir}\n"
        "============================================================\n"
    )


if __name__ == "__main__":
    main()