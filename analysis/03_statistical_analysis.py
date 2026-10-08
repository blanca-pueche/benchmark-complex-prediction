#!/usr/bin/env python3

"""
03_statistical_analysis.py

Inferential statistical analysis of the biomolecular structure
prediction benchmark.

Input
-----
benchmark_dataset.csv

Outputs
-------
<output_dir>/
    tables/
        friedman_tests.csv
        pairwise_wilcoxon.csv
        effect_sizes.csv

        interface_friedman_tests.csv
        interface_pairwise_wilcoxon.csv
        interface_effect_sizes.csv

        success_rate_mcnemar.csv
        runtime_friedman.csv
        runtime_pairwise_wilcoxon.csv

        category_friedman.csv
        category_pairwise_wilcoxon.csv

        category_interface_friedman.csv
        category_interface_pairwise_wilcoxon.csv
        category_interface_effect_sizes.csv

    figures/
        01_effect_size_lddt.png
        02_effect_size_tm_score.png
        03_effect_size_rmsd.png
        04_effect_size_dockq.png
        05_success_rate_comparison.png

        06_effect_size_qs_global.png
        07_effect_size_gdt_ts.png
        08_effect_size_gdt_ha.png

    statistical_results.txt

Statistical framework
---------------------
- Predictions are paired by benchmark structure.
- Accuracy metrics:
    Friedman test for global comparison.
    Paired Wilcoxon signed-rank tests for post-hoc comparisons.
    Benjamini-Hochberg FDR correction.
    Rank-biserial correlation as paired effect size.

- Primary structural metrics:
    lDDT
    TM-score
    RMSD
    DockQ

- Interface/quaternary metrics:
    QS-score
    GDT-TS
    GDT-HA

  DockQ is not repeated in this analysis because it is already
  included in the primary accuracy analysis.

- Success/failure:
    McNemar's test for pairwise comparison of success rates.
    Benjamini-Hochberg FDR correction.

- Runtime:
    AF3 excluded because predictions are precomputed.
    Friedman test followed by paired Wilcoxon tests.

- Failed predictions are never assigned an accuracy score of zero.
- Complete paired cases are used for accuracy comparisons.
- Category-specific analyses are reported separately.
"""

import argparse
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    chi2,
)

from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")


# ============================================================
# PREDICTORS AND CATEGORIES
# ============================================================

PREDICTORS = [
    "AF3",
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold",
]

CATEGORIES = [
    "Protein-Protein",
    "Protein-Antibody",
    "Protein-DNA",
    "Protein-RNA",
    "Protein-Peptide",
]


# ============================================================
# METRICS
# ============================================================

# Primary metrics used for the main overall accuracy analysis.
PRIMARY_METRICS = [
    "lddt",
    "tm_score",
    "rmsd",
    "dockq_ave",
]


# Additional metrics used specifically for Section 3.4.
# DockQ is intentionally not repeated here because it is
# already analyzed as a primary metric.
INTERFACE_METRICS = [
    "qs_global",
    "oligo_gdtts",
    "oligo_gdtha",
]


INTERFACE_METRIC_LABELS = {
    "qs_global": "QS-score",
    "oligo_gdtts": "GDT-TS",
    "oligo_gdtha": "GDT-HA",
}


HIGHER_IS_BETTER = {
    "lddt": True,
    "tm_score": True,
    "rmsd": False,
    "dockq_ave": True,
    "qs_global": True,
    "oligo_gdtts": True,
    "oligo_gdtha": True,
}


# All metrics that need numeric conversion.
ALL_ACCURACY_METRICS = list(
    dict.fromkeys(
        PRIMARY_METRICS
        + INTERFACE_METRICS
    )
)


# ============================================================
# HELPER: PAIRED DATA
# ============================================================

def get_paired_data(
    data,
    metric,
    predictors=PREDICTORS
):
    """
    Return a structure x predictor matrix containing only
    complete paired observations for the requested metric.

    A structure is included only if all predictors have a
    valid metric value.
    """

    if metric not in data.columns:
        return pd.DataFrame(
            columns=predictors
        )

    subset = data[
        data["predictor"].isin(predictors)
    ].copy()

    matrix = subset.pivot(
        index="pdb_id",
        columns="predictor",
        values=metric
    )

    matrix = matrix.reindex(
        columns=predictors
    )

    matrix = matrix.dropna(
        how="any"
    )

    return matrix


# ============================================================
# HELPER: EFFECT SIZE
# ============================================================

def rank_biserial_correlation(
    x,
    y
):
    """
    Calculate rank-biserial correlation for paired data.

    Positive values indicate that x tends to be greater
    than y.

    Range:
        -1 to +1
    """

    differences = (
        np.asarray(x)
        - np.asarray(y)
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    abs_diff = np.abs(
        differences
    )

    ranks = pd.Series(
        abs_diff
    ).rank(
        method="average"
    ).values

    positive_ranks = ranks[
        differences > 0
    ].sum()

    negative_ranks = ranks[
        differences < 0
    ].sum()

    total = (
        positive_ranks
        + negative_ranks
    )

    if total == 0:
        return 0.0

    return (
        positive_ranks
        - negative_ranks
    ) / total


# ============================================================
# HELPER: EFFECT INTERPRETATION
# ============================================================

def interpret_effect(effect):

    absolute = abs(effect)

    if absolute < 0.1:
        return "negligible"

    if absolute < 0.3:
        return "small"

    if absolute < 0.5:
        return "moderate"

    return "large"


# ============================================================
# HELPER: MCMEMAR TEST
# ============================================================

def mcnemar_test(
    x,
    y
):
    """
    McNemar test using the continuity-corrected
    chi-square approximation.

    x and y are binary paired outcomes.
    """

    x = np.asarray(x).astype(bool)
    y = np.asarray(y).astype(bool)

    b = np.sum(
        (x == True)
        & (y == False)
    )

    c = np.sum(
        (x == False)
        & (y == True)
    )

    if b + c == 0:
        return 0.0, 1.0, b, c

    statistic = (
        (abs(b - c) - 1) ** 2
        / (b + c)
    )

    p_value = 1 - chi2.cdf(
        statistic,
        df=1
    )

    return (
        statistic,
        p_value,
        b,
        c
    )


# ============================================================
# HELPER: EFFECT SIZE HEATMAP
# ============================================================

def make_effect_heatmap(
    effect_data,
    metric,
    filename,
    title,
    figure_dir
):
    """
    Create a predictor x predictor heatmap of
    rank-biserial effect sizes.

    Positive values mean the predictor in the row tends
    to have higher metric values than the predictor
    in the column.
    """

    matrix = pd.DataFrame(
        np.zeros(
            (
                len(PREDICTORS),
                len(PREDICTORS)
            )
        ),
        index=PREDICTORS,
        columns=PREDICTORS
    )

    if effect_data.empty:
        return

    metric_data = effect_data[
        effect_data["metric"] == metric
    ]

    for _, row in metric_data.iterrows():

        p1 = row[
            "predictor_1"
        ]

        p2 = row[
            "predictor_2"
        ]

        effect = row[
            "rank_biserial_correlation"
        ]

        matrix.loc[
            p1,
            p2
        ] = effect

        matrix.loc[
            p2,
            p1
        ] = -effect

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im = ax.imshow(
        matrix.values,
        vmin=-1,
        vmax=1,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(len(PREDICTORS))
    )

    ax.set_yticks(
        np.arange(len(PREDICTORS))
    )

    ax.set_xticklabels(
        PREDICTORS,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        PREDICTORS
    )

    ax.set_xlabel(
        "Comparison predictor"
    )

    ax.set_ylabel(
        "Reference predictor"
    )

    ax.set_title(
        title
    )

    for i in range(
        len(PREDICTORS)
    ):

        for j in range(
            len(PREDICTORS)
        ):

            if i == j:
                continue

            value = matrix.iloc[
                i,
                j
            ]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Rank-biserial correlation"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figure_dir,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Perform inferential statistical analysis of the "
            "biomolecular structure prediction benchmark."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Input benchmark_dataset.csv generated by "
            "01_prepare_dataset.py."
        )
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=(
            "Output directory where statistical results "
            "will be saved."
        )
    )

    args = parser.parse_args()

    input_file = os.path.abspath(
        os.path.expanduser(
            args.input
        )
    )

    output_dir = os.path.abspath(
        os.path.expanduser(
            args.output_dir
        )
    )

    table_dir = os.path.join(
        output_dir,
        "tables"
    )

    figure_dir = os.path.join(
        output_dir,
        "figures"
    )

    os.makedirs(
        table_dir,
        exist_ok=True
    )

    os.makedirs(
        figure_dir,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("=" * 70)
    print("LOADING BENCHMARK DATASET")
    print("=" * 70)

    if not os.path.exists(input_file):

        raise FileNotFoundError(
            f"Could not find:\n{input_file}\n\n"
            "Run 01_prepare_dataset.py first."
        )

    df = pd.read_csv(
        input_file
    )

    print(
        f"Rows:       {len(df)}"
    )

    print(
        f"Structures: {df['pdb_id'].nunique()}"
    )

    print()

    # ========================================================
    # CLEAN DATA
    # ========================================================

    df["pdb_id"] = (
        df["pdb_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["predictor"] = (
        df["predictor"]
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
    )

    # Convert all accuracy metrics to numeric.
    for metric in ALL_ACCURACY_METRICS:

        if metric in df.columns:

            df[metric] = pd.to_numeric(
                df[metric],
                errors="coerce"
            )

    if "prediction_success" in df.columns:

        df["prediction_success"] = (
            df["prediction_success"]
            .astype(str)
            .str.lower()
            .isin([
                "true",
                "1",
                "yes"
            ])
        )

    else:

        df["prediction_success"] = (
            df["status"]
            .astype(str)
            .str.upper()
            == "SUCCESS"
        )

    # ========================================================
    # ENSURE ONE ROW PER STRUCTURE / PREDICTOR
    # ========================================================

    df = (
        df
        .drop_duplicates(
            subset=[
                "pdb_id",
                "predictor"
            ]
        )
        .copy()
    )

    # ========================================================
    # PRIMARY FRIEDMAN TEST
    # ========================================================

    print("=" * 70)
    print("FRIEDMAN TESTS")
    print("=" * 70)

    friedman_rows = []

    for metric in PRIMARY_METRICS:

        matrix = get_paired_data(
            df,
            metric
        )

        n = len(matrix)

        if n < 3:

            print(
                f"{metric}: insufficient paired observations"
            )

            continue

        values = [
            matrix[predictor].values
            for predictor in PREDICTORS
        ]

        statistic, p_value = friedmanchisquare(
            *values
        )

        friedman_rows.append({

            "metric": metric,

            "n_complete_structures": n,

            "n_predictors": len(PREDICTORS),

            "friedman_chi_square": statistic,

            "p_value": p_value,

            "significant_alpha_0.05": (
                p_value < 0.05
            ),

        })

        print(
            f"{metric:15s} "
            f"n={n:3d}  "
            f"chi2={statistic:.3f}  "
            f"p={p_value:.4g}"
        )

    friedman_table = pd.DataFrame(
        friedman_rows
    )

    friedman_table.to_csv(
        os.path.join(
            table_dir,
            "friedman_tests.csv"
        ),
        index=False
    )

    # ========================================================
    # PRIMARY PAIRWISE WILCOXON TESTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PAIRWISE WILCOXON TESTS")
    print("=" * 70)

    wilcoxon_rows = []

    for metric in PRIMARY_METRICS:

        matrix = get_paired_data(
            df,
            metric
        )

        if len(matrix) < 3:
            continue

        for predictor_a, predictor_b in combinations(
            PREDICTORS,
            2
        ):

            x = matrix[
                predictor_a
            ].values

            y = matrix[
                predictor_b
            ].values

            differences = x - y

            nonzero = (
                differences != 0
            )

            x_test = x[
                nonzero
            ]

            y_test = y[
                nonzero
            ]

            n_nonzero = len(
                x_test
            )

            if n_nonzero == 0:

                statistic = 0.0
                p_value = 1.0
                effect = 0.0

            else:

                try:

                    statistic, p_value = wilcoxon(
                        x_test,
                        y_test,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )

                except TypeError:

                    statistic, p_value = wilcoxon(
                        x_test,
                        y_test,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )

                effect = rank_biserial_correlation(
                    x_test,
                    y_test
                )

            median_a = np.median(x)
            median_b = np.median(y)

            median_difference = (
                median_a
                - median_b
            )

            wilcoxon_rows.append({

                "metric": metric,

                "predictor_1": predictor_a,

                "predictor_2": predictor_b,

                "n_complete": len(matrix),

                "n_nonzero_differences": n_nonzero,

                "median_predictor_1": median_a,

                "median_predictor_2": median_b,

                "median_difference": median_difference,

                "wilcoxon_statistic": statistic,

                "p_value": p_value,

                "rank_biserial_correlation": effect,

                "effect_interpretation": interpret_effect(
                    effect
                ),

            })

    wilcoxon_table = pd.DataFrame(
        wilcoxon_rows
    )

    # ========================================================
    # PRIMARY FDR CORRECTION
    # ========================================================

    if not wilcoxon_table.empty:

        wilcoxon_table[
            "p_value_fdr"
        ] = np.nan

        wilcoxon_table[
            "significant_fdr"
        ] = False

        for metric in PRIMARY_METRICS:

            mask = (
                wilcoxon_table["metric"]
                == metric
            )

            p_values = (
                wilcoxon_table
                .loc[mask, "p_value"]
                .values
            )

            if len(p_values) == 0:
                continue

            rejected, corrected, _, _ = (
                multipletests(
                    p_values,
                    alpha=0.05,
                    method="fdr_bh"
                )
            )

            wilcoxon_table.loc[
                mask,
                "p_value_fdr"
            ] = corrected

            wilcoxon_table.loc[
                mask,
                "significant_fdr"
            ] = rejected

    wilcoxon_table.to_csv(
        os.path.join(
            table_dir,
            "pairwise_wilcoxon.csv"
        ),
        index=False
    )

    # ========================================================
    # PRIMARY EFFECT SIZE TABLE
    # ========================================================

    effect_table = wilcoxon_table[
        [
            "metric",
            "predictor_1",
            "predictor_2",
            "n_complete",
            "median_difference",
            "rank_biserial_correlation",
            "effect_interpretation",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
        ]
    ].copy()

    effect_table.to_csv(
        os.path.join(
            table_dir,
            "effect_sizes.csv"
        ),
        index=False
    )

    # ========================================================
    # PRINT SIGNIFICANT PRIMARY COMPARISONS
    # ========================================================

    print(
        "\nSignificant primary-metric pairwise comparisons "
        "after FDR correction:"
    )

    if not wilcoxon_table.empty:

        significant = wilcoxon_table[
            wilcoxon_table[
                "significant_fdr"
            ]
        ]

        if significant.empty:

            print(
                "  None"
            )

        else:

            for _, row in significant.iterrows():

                print(
                    f"  {row['metric']}: "
                    f"{row['predictor_1']} vs "
                    f"{row['predictor_2']} "
                    f"(FDR p={row['p_value_fdr']:.4g}, "
                    f"effect={row['rank_biserial_correlation']:.3f})"
                )

    # ========================================================
    # INTERFACE / QUATERNARY FRIEDMAN TESTS
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERFACE / QUATERNARY FRIEDMAN TESTS")
    print("=" * 70)

    interface_friedman_rows = []

    for metric in INTERFACE_METRICS:

        if metric not in df.columns:

            print(
                f"{metric}: column not available"
            )

            continue

        matrix = get_paired_data(
            df,
            metric
        )

        n = len(matrix)

        if n < 3:

            print(
                f"{metric}: insufficient paired observations"
            )

            continue

        values = [
            matrix[predictor].values
            for predictor in PREDICTORS
        ]

        statistic, p_value = friedmanchisquare(
            *values
        )

        interface_friedman_rows.append({

            "metric": metric,

            "metric_label": INTERFACE_METRIC_LABELS[
                metric
            ],

            "n_complete_structures": n,

            "n_predictors": len(PREDICTORS),

            "friedman_chi_square": statistic,

            "p_value": p_value,

            "significant_alpha_0.05": (
                p_value < 0.05
            ),

        })

        print(
            f"{INTERFACE_METRIC_LABELS[metric]:15s} "
            f"n={n:3d}  "
            f"chi2={statistic:.3f}  "
            f"p={p_value:.4g}"
        )

    interface_friedman_table = pd.DataFrame(
        interface_friedman_rows
    )

    interface_friedman_table.to_csv(
        os.path.join(
            table_dir,
            "interface_friedman_tests.csv"
        ),
        index=False
    )

    # ========================================================
    # INTERFACE / QUATERNARY PAIRWISE WILCOXON TESTS
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERFACE / QUATERNARY PAIRWISE WILCOXON TESTS")
    print("=" * 70)

    interface_wilcoxon_rows = []

    for metric in INTERFACE_METRICS:

        if metric not in df.columns:
            continue

        matrix = get_paired_data(
            df,
            metric
        )

        if len(matrix) < 3:
            continue

        for predictor_a, predictor_b in combinations(
            PREDICTORS,
            2
        ):

            x = matrix[
                predictor_a
            ].values

            y = matrix[
                predictor_b
            ].values

            differences = x - y

            nonzero = (
                differences != 0
            )

            x_test = x[
                nonzero
            ]

            y_test = y[
                nonzero
            ]

            n_nonzero = len(
                x_test
            )

            if n_nonzero == 0:

                statistic = 0.0
                p_value = 1.0
                effect = 0.0

            else:

                try:

                    statistic, p_value = wilcoxon(
                        x_test,
                        y_test,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )

                except TypeError:

                    statistic, p_value = wilcoxon(
                        x_test,
                        y_test,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )

                effect = rank_biserial_correlation(
                    x_test,
                    y_test
                )

            median_a = np.median(x)
            median_b = np.median(y)

            interface_wilcoxon_rows.append({

                "metric": metric,

                "metric_label": INTERFACE_METRIC_LABELS[
                    metric
                ],

                "predictor_1": predictor_a,

                "predictor_2": predictor_b,

                "n_complete": len(matrix),

                "n_nonzero_differences": n_nonzero,

                "median_predictor_1": median_a,

                "median_predictor_2": median_b,

                "median_difference": (
                    median_a
                    - median_b
                ),

                "wilcoxon_statistic": statistic,

                "p_value": p_value,

                "rank_biserial_correlation": effect,

                "effect_interpretation": interpret_effect(
                    effect
                ),

            })

    interface_wilcoxon_table = pd.DataFrame(
        interface_wilcoxon_rows
    )

    # ========================================================
    # INTERFACE FDR CORRECTION
    # ========================================================

    if not interface_wilcoxon_table.empty:

        interface_wilcoxon_table[
            "p_value_fdr"
        ] = np.nan

        interface_wilcoxon_table[
            "significant_fdr"
        ] = False

        for metric in INTERFACE_METRICS:

            mask = (
                interface_wilcoxon_table["metric"]
                == metric
            )

            p_values = (
                interface_wilcoxon_table
                .loc[
                    mask,
                    "p_value"
                ]
                .values
            )

            if len(p_values) == 0:
                continue

            rejected, corrected, _, _ = (
                multipletests(
                    p_values,
                    alpha=0.05,
                    method="fdr_bh"
                )
            )

            interface_wilcoxon_table.loc[
                mask,
                "p_value_fdr"
            ] = corrected

            interface_wilcoxon_table.loc[
                mask,
                "significant_fdr"
            ] = rejected

    interface_wilcoxon_table.to_csv(
        os.path.join(
            table_dir,
            "interface_pairwise_wilcoxon.csv"
        ),
        index=False
    )

    # ========================================================
    # INTERFACE EFFECT SIZE TABLE
    # ========================================================

    interface_effect_table = interface_wilcoxon_table[
        [
            "metric",
            "metric_label",
            "predictor_1",
            "predictor_2",
            "n_complete",
            "median_difference",
            "rank_biserial_correlation",
            "effect_interpretation",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
        ]
    ].copy()

    interface_effect_table.to_csv(
        os.path.join(
            table_dir,
            "interface_effect_sizes.csv"
        ),
        index=False
    )

    # ========================================================
    # PRINT SIGNIFICANT INTERFACE COMPARISONS
    # ========================================================

    print(
        "\nSignificant interface/quaternary pairwise comparisons "
        "after FDR correction:"
    )

    if not interface_wilcoxon_table.empty:

        significant_interface = (
            interface_wilcoxon_table[
                interface_wilcoxon_table[
                    "significant_fdr"
                ]
            ]
        )

        if significant_interface.empty:

            print(
                "  None"
            )

        else:

            for _, row in (
                significant_interface.iterrows()
            ):

                print(
                    f"  {row['metric_label']}: "
                    f"{row['predictor_1']} vs "
                    f"{row['predictor_2']} "
                    f"(FDR p={row['p_value_fdr']:.4g}, "
                    f"effect={row['rank_biserial_correlation']:.3f})"
                )

    # ========================================================
    # MCMEMAR TEST FOR SUCCESS RATE
    # ========================================================

    print("\n" + "=" * 70)
    print("SUCCESS-RATE COMPARISONS")
    print("=" * 70)

    success_matrix = df.pivot(
        index="pdb_id",
        columns="predictor",
        values="prediction_success"
    )

    success_matrix = success_matrix.reindex(
        columns=PREDICTORS
    )

    success_rows = []

    for predictor_a, predictor_b in combinations(
        PREDICTORS,
        2
    ):

        paired = success_matrix[
            [
                predictor_a,
                predictor_b
            ]
        ].dropna()

        if paired.empty:
            continue

        x = paired[
            predictor_a
        ].values

        y = paired[
            predictor_b
        ].values

        statistic, p_value, b, c = (
            mcnemar_test(
                x,
                y
            )
        )

        success_a = np.sum(x)
        success_b = np.sum(y)

        n = len(paired)

        success_rows.append({

            "predictor_1": predictor_a,

            "predictor_2": predictor_b,

            "n_structures": n,

            "success_1": int(success_a),

            "success_2": int(success_b),

            "success_rate_1_percent": (
                100 * success_a / n
            ),

            "success_rate_2_percent": (
                100 * success_b / n
            ),

            "discordant_1_success_2_fail": int(b),

            "discordant_1_fail_2_success": int(c),

            "mcnemar_chi_square": statistic,

            "p_value": p_value,

        })

    success_table = pd.DataFrame(
        success_rows
    )

    if not success_table.empty:

        rejected, corrected, _, _ = (
            multipletests(
                success_table["p_value"],
                alpha=0.05,
                method="fdr_bh"
            )
        )

        success_table[
            "p_value_fdr"
        ] = corrected

        success_table[
            "significant_fdr"
        ] = rejected

    success_table.to_csv(
        os.path.join(
            table_dir,
            "success_rate_mcnemar.csv"
        ),
        index=False
    )

    # ========================================================
    # RUNTIME ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNTIME ANALYSIS")
    print("=" * 70)

    runtime_friedman = None
    runtime_pairwise = pd.DataFrame()

    if "execution_seconds" in df.columns:

        runtime = df[
            (df["prediction_success"])
            &
            (df["execution_seconds"].notna())
            &
            (df["predictor"] != "AF3")
        ].copy()

        runtime_matrix = runtime.pivot(
            index="pdb_id",
            columns="predictor",
            values="execution_seconds"
        )

        runtime_predictors = [
            predictor
            for predictor in PREDICTORS
            if predictor != "AF3"
            and predictor in runtime_matrix.columns
        ]

        runtime_matrix = runtime_matrix.reindex(
            columns=runtime_predictors
        )

        runtime_matrix = runtime_matrix.dropna(
            how="any"
        )

        if (
            len(runtime_predictors) >= 3
            and len(runtime_matrix) >= 3
        ):

            runtime_values = [
                runtime_matrix[predictor].values
                for predictor in runtime_predictors
            ]

            statistic, p_value = (
                friedmanchisquare(
                    *runtime_values
                )
            )

            runtime_friedman = pd.DataFrame([
                {
                    "n_complete_structures": len(
                        runtime_matrix
                    ),
                    "predictors": ", ".join(
                        runtime_predictors
                    ),
                    "friedman_chi_square": statistic,
                    "p_value": p_value,
                }
            ])

            runtime_friedman.to_csv(
                os.path.join(
                    table_dir,
                    "runtime_friedman.csv"
                ),
                index=False
            )

            runtime_pairwise_rows = []

            for predictor_a, predictor_b in combinations(
                runtime_predictors,
                2
            ):

                x = runtime_matrix[
                    predictor_a
                ].values

                y = runtime_matrix[
                    predictor_b
                ].values

                try:

                    statistic, p_value = wilcoxon(
                        x,
                        y,
                        alternative="two-sided"
                    )

                except TypeError:

                    statistic, p_value = wilcoxon(
                        x,
                        y,
                        alternative="two-sided"
                    )

                effect = rank_biserial_correlation(
                    x,
                    y
                )

                runtime_pairwise_rows.append({

                    "predictor_1": predictor_a,

                    "predictor_2": predictor_b,

                    "n_complete": len(
                        runtime_matrix
                    ),

                    "median_runtime_1_seconds": np.median(x),

                    "median_runtime_2_seconds": np.median(y),

                    "median_difference_seconds": (
                        np.median(x)
                        - np.median(y)
                    ),

                    "wilcoxon_statistic": statistic,

                    "p_value": p_value,

                    "rank_biserial_correlation": effect,

                })

            runtime_pairwise = pd.DataFrame(
                runtime_pairwise_rows
            )

            if not runtime_pairwise.empty:

                rejected, corrected, _, _ = (
                    multipletests(
                        runtime_pairwise["p_value"],
                        alpha=0.05,
                        method="fdr_bh"
                    )
                )

                runtime_pairwise[
                    "p_value_fdr"
                ] = corrected

                runtime_pairwise[
                    "significant_fdr"
                ] = rejected

            runtime_pairwise.to_csv(
                os.path.join(
                    table_dir,
                    "runtime_pairwise_wilcoxon.csv"
                ),
                index=False
            )

            print(
                f"Complete paired runtime structures: "
                f"{len(runtime_matrix)}"
            )

            print(
                f"Friedman p-value: "
                f"{p_value:.4g}"
            )

        else:

            print(
                "Insufficient paired runtime data."
            )

    else:

        print(
            "execution_seconds column not available."
        )

    # ========================================================
    # CATEGORY-SPECIFIC PRIMARY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("CATEGORY-SPECIFIC PRIMARY STATISTICAL ANALYSIS")
    print("=" * 70)

    category_friedman_rows = []
    category_pairwise_rows = []

    for category in CATEGORIES:

        category_data = df[
            df["category"] == category
        ].copy()

        print(
            f"\n{category}"
        )

        for metric in PRIMARY_METRICS:

            matrix = get_paired_data(
                category_data,
                metric
            )

            n = len(matrix)

            if n < 3:
                continue

            values = [
                matrix[predictor].values
                for predictor in PREDICTORS
            ]

            try:

                statistic, p_value = (
                    friedmanchisquare(
                        *values
                    )
                )

            except ValueError:

                continue

            category_friedman_rows.append({

                "category": category,

                "metric": metric,

                "n_complete_structures": n,

                "friedman_chi_square": statistic,

                "p_value": p_value,

            })

            print(
                f"  {metric:15s} "
                f"n={n:3d} "
                f"p={p_value:.4g}"
            )

            for predictor_a, predictor_b in combinations(
                PREDICTORS,
                2
            ):

                x = matrix[
                    predictor_a
                ].values

                y = matrix[
                    predictor_b
                ].values

                differences = x - y

                nonzero = (
                    differences != 0
                )

                x_test = x[
                    nonzero
                ]

                y_test = y[
                    nonzero
                ]

                if len(x_test) == 0:

                    statistic_w = 0.0
                    p_w = 1.0
                    effect = 0.0

                else:

                    try:

                        statistic_w, p_w = wilcoxon(
                            x_test,
                            y_test,
                            alternative="two-sided"
                        )

                    except TypeError:

                        statistic_w, p_w = wilcoxon(
                            x_test,
                            y_test,
                            alternative="two-sided"
                        )

                    effect = rank_biserial_correlation(
                        x_test,
                        y_test
                    )

                category_pairwise_rows.append({

                    "category": category,

                    "metric": metric,

                    "predictor_1": predictor_a,

                    "predictor_2": predictor_b,

                    "n_complete": n,

                    "median_difference": (
                        np.median(x)
                        - np.median(y)
                    ),

                    "wilcoxon_statistic": statistic_w,

                    "p_value": p_w,

                    "rank_biserial_correlation": effect,

                    "effect_interpretation": interpret_effect(
                        effect
                    ),

                })

    category_friedman = pd.DataFrame(
        category_friedman_rows
    )

    category_pairwise = pd.DataFrame(
        category_pairwise_rows
    )

    # ========================================================
    # CATEGORY PRIMARY FDR CORRECTION
    # ========================================================

    if not category_pairwise.empty:

        category_pairwise[
            "p_value_fdr"
        ] = np.nan

        category_pairwise[
            "significant_fdr"
        ] = False

        for category in CATEGORIES:

            for metric in PRIMARY_METRICS:

                mask = (
                    (category_pairwise["category"] == category)
                    &
                    (category_pairwise["metric"] == metric)
                )

                p_values = (
                    category_pairwise
                    .loc[mask, "p_value"]
                    .values
                )

                if len(p_values) == 0:
                    continue

                rejected, corrected, _, _ = (
                    multipletests(
                        p_values,
                        alpha=0.05,
                        method="fdr_bh"
                    )
                )

                category_pairwise.loc[
                    mask,
                    "p_value_fdr"
                ] = corrected

                category_pairwise.loc[
                    mask,
                    "significant_fdr"
                ] = rejected

    category_friedman.to_csv(
        os.path.join(
            table_dir,
            "category_friedman.csv"
        ),
        index=False
    )

    category_pairwise.to_csv(
        os.path.join(
            table_dir,
            "category_pairwise_wilcoxon.csv"
        ),
        index=False
    )

    # ========================================================
    # CATEGORY-SPECIFIC INTERFACE / QUATERNARY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("CATEGORY-SPECIFIC INTERFACE / QUATERNARY ANALYSIS")
    print("=" * 70)

    category_interface_friedman_rows = []
    category_interface_pairwise_rows = []

    for category in CATEGORIES:

        category_data = df[
            df["category"] == category
        ].copy()

        print(
            f"\n{category}"
        )

        for metric in INTERFACE_METRICS:

            if metric not in category_data.columns:
                continue

            matrix = get_paired_data(
                category_data,
                metric
            )

            n = len(matrix)

            if n < 3:
                continue

            values = [
                matrix[predictor].values
                for predictor in PREDICTORS
            ]

            try:

                statistic, p_value = (
                    friedmanchisquare(
                        *values
                    )
                )

            except ValueError:

                continue

            category_interface_friedman_rows.append({

                "category": category,

                "metric": metric,

                "metric_label": INTERFACE_METRIC_LABELS[
                    metric
                ],

                "n_complete_structures": n,

                "friedman_chi_square": statistic,

                "p_value": p_value,

            })

            print(
                f"  {INTERFACE_METRIC_LABELS[metric]:15s} "
                f"n={n:3d} "
                f"p={p_value:.4g}"
            )

            for predictor_a, predictor_b in combinations(
                PREDICTORS,
                2
            ):

                x = matrix[
                    predictor_a
                ].values

                y = matrix[
                    predictor_b
                ].values

                differences = x - y

                nonzero = (
                    differences != 0
                )

                x_test = x[
                    nonzero
                ]

                y_test = y[
                    nonzero
                ]

                if len(x_test) == 0:

                    statistic_w = 0.0
                    p_w = 1.0
                    effect = 0.0

                else:

                    try:

                        statistic_w, p_w = wilcoxon(
                            x_test,
                            y_test,
                            alternative="two-sided"
                        )

                    except TypeError:

                        statistic_w, p_w = wilcoxon(
                            x_test,
                            y_test,
                            alternative="two-sided"
                        )

                    effect = rank_biserial_correlation(
                        x_test,
                        y_test
                    )

                category_interface_pairwise_rows.append({

                    "category": category,

                    "metric": metric,

                    "metric_label": INTERFACE_METRIC_LABELS[
                        metric
                    ],

                    "predictor_1": predictor_a,

                    "predictor_2": predictor_b,

                    "n_complete": n,

                    "median_predictor_1": np.median(x),

                    "median_predictor_2": np.median(y),

                    "median_difference": (
                        np.median(x)
                        - np.median(y)
                    ),

                    "wilcoxon_statistic": statistic_w,

                    "p_value": p_w,

                    "rank_biserial_correlation": effect,

                    "effect_interpretation": interpret_effect(
                        effect
                    ),

                })

    category_interface_friedman = pd.DataFrame(
        category_interface_friedman_rows
    )

    category_interface_pairwise = pd.DataFrame(
        category_interface_pairwise_rows
    )

    # ========================================================
    # CATEGORY INTERFACE FDR CORRECTION
    # ========================================================

    if not category_interface_pairwise.empty:

        category_interface_pairwise[
            "p_value_fdr"
        ] = np.nan

        category_interface_pairwise[
            "significant_fdr"
        ] = False

        for category in CATEGORIES:

            for metric in INTERFACE_METRICS:

                mask = (
                    (category_interface_pairwise["category"] == category)
                    &
                    (category_interface_pairwise["metric"] == metric)
                )

                p_values = (
                    category_interface_pairwise
                    .loc[
                        mask,
                        "p_value"
                    ]
                    .values
                )

                if len(p_values) == 0:
                    continue

                rejected, corrected, _, _ = (
                    multipletests(
                        p_values,
                        alpha=0.05,
                        method="fdr_bh"
                    )
                )

                category_interface_pairwise.loc[
                    mask,
                    "p_value_fdr"
                ] = corrected

                category_interface_pairwise.loc[
                    mask,
                    "significant_fdr"
                ] = rejected

    category_interface_pairwise.to_csv(
        os.path.join(
            table_dir,
            "category_interface_pairwise_wilcoxon.csv"
        ),
        index=False
    )

    category_interface_friedman.to_csv(
        os.path.join(
            table_dir,
            "category_interface_friedman.csv"
        ),
        index=False
    )

    # ========================================================
    # CATEGORY INTERFACE EFFECT SIZE TABLE
    # ========================================================

    if not category_interface_pairwise.empty:

        category_interface_effect_table = (
            category_interface_pairwise[
                [
                    "category",
                    "metric",
                    "metric_label",
                    "predictor_1",
                    "predictor_2",
                    "n_complete",
                    "median_difference",
                    "rank_biserial_correlation",
                    "effect_interpretation",
                    "p_value",
                    "p_value_fdr",
                    "significant_fdr",
                ]
            ].copy()
        )

    else:

        category_interface_effect_table = pd.DataFrame()

    category_interface_effect_table.to_csv(
        os.path.join(
            table_dir,
            "category_interface_effect_sizes.csv"
        ),
        index=False
    )

    # ========================================================
    # PRIMARY EFFECT SIZE FIGURES
    # ========================================================

    if not effect_table.empty:

        make_effect_heatmap(
            effect_table,
            "lddt",
            "01_effect_size_lddt.png",
            "Pairwise effect sizes: lDDT",
            figure_dir
        )

        make_effect_heatmap(
            effect_table,
            "tm_score",
            "02_effect_size_tm_score.png",
            "Pairwise effect sizes: TM-score",
            figure_dir
        )

        make_effect_heatmap(
            effect_table,
            "rmsd",
            "03_effect_size_rmsd.png",
            "Pairwise effect sizes: RMSD",
            figure_dir
        )

        make_effect_heatmap(
            effect_table,
            "dockq_ave",
            "04_effect_size_dockq.png",
            "Pairwise effect sizes: DockQ",
            figure_dir
        )

    # ========================================================
    # SUCCESS RATE FIGURE
    # ========================================================

    success_rates = (
        df
        .groupby(
            "predictor",
            observed=True
        )["prediction_success"]
        .mean()
        .reindex(PREDICTORS)
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    x = np.arange(
        len(PREDICTORS)
    )

    bars = ax.bar(
        x,
        success_rates.values
    )

    ax.set_xticks(x)
    ax.set_xticklabels(PREDICTORS)

    ax.set_ylabel(
        "Successful predictions (%)"
    )

    ax.set_xlabel(
        "Prediction method"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "Prediction success rate"
    )

    for bar, value in zip(
        bars,
        success_rates.values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figure_dir,
            "05_success_rate_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # INTERFACE EFFECT SIZE FIGURES
    # ========================================================

    if not interface_effect_table.empty:

        make_effect_heatmap(
            interface_effect_table,
            "qs_global",
            "06_effect_size_qs_global.png",
            "Pairwise effect sizes: QS-score",
            figure_dir
        )

        make_effect_heatmap(
            interface_effect_table,
            "oligo_gdtts",
            "07_effect_size_gdt_ts.png",
            "Pairwise effect sizes: GDT-TS",
            figure_dir
        )

        make_effect_heatmap(
            interface_effect_table,
            "oligo_gdtha",
            "08_effect_size_gdt_ha.png",
            "Pairwise effect sizes: GDT-HA",
            figure_dir
        )

    # ========================================================
    # STATISTICAL RESULTS TEXT FILE
    # ========================================================

    stats_file = os.path.join(
        output_dir,
        "statistical_results.txt"
    )

    with open(
        stats_file,
        "w"
    ) as f:

        f.write(
            "BIOMOLECULAR STRUCTURE PREDICTION BENCHMARK\n"
        )

        f.write(
            "STATISTICAL ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Total structures: "
            f"{df['pdb_id'].nunique()}\n"
        )

        f.write(
            f"Predictors: "
            f"{', '.join(PREDICTORS)}\n\n"
        )

        # ----------------------------------------------------
        # Primary Friedman
        # ----------------------------------------------------

        f.write(
            "GLOBAL FRIEDMAN TESTS ? PRIMARY METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in friedman_table.iterrows():

            f.write(
                f"{row['metric']}: "
                f"n={int(row['n_complete_structures'])}, "
                f"chi2={row['friedman_chi_square']:.4f}, "
                f"p={row['p_value']:.6g}\n"
            )

        # ----------------------------------------------------
        # Primary significant pairwise
        # ----------------------------------------------------

        f.write(
            "\nSIGNIFICANT PRIMARY PAIRWISE COMPARISONS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        significant = wilcoxon_table[
            wilcoxon_table[
                "significant_fdr"
            ]
        ]

        if significant.empty:

            f.write(
                "No significant pairwise comparisons "
                "after FDR correction.\n"
            )

        else:

            for _, row in significant.iterrows():

                f.write(
                    f"{row['metric']}: "
                    f"{row['predictor_1']} vs "
                    f"{row['predictor_2']}; "
                    f"FDR p={row['p_value_fdr']:.6g}; "
                    f"effect="
                    f"{row['rank_biserial_correlation']:.3f}; "
                    f"{row['effect_interpretation']}\n"
                )

        # ----------------------------------------------------
        # Interface Friedman
        # ----------------------------------------------------

        f.write(
            "\nGLOBAL FRIEDMAN TESTS ? "
            "INTERFACE / QUATERNARY METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in interface_friedman_table.iterrows():

            f.write(
                f"{row['metric_label']}: "
                f"n={int(row['n_complete_structures'])}, "
                f"chi2={row['friedman_chi_square']:.4f}, "
                f"p={row['p_value']:.6g}\n"
            )

        # ----------------------------------------------------
        # Interface significant pairwise
        # ----------------------------------------------------

        f.write(
            "\nSIGNIFICANT INTERFACE / QUATERNARY "
            "PAIRWISE COMPARISONS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        significant_interface = (
            interface_wilcoxon_table[
                interface_wilcoxon_table[
                    "significant_fdr"
                ]
            ]
        )

        if significant_interface.empty:

            f.write(
                "No significant pairwise comparisons "
                "after FDR correction.\n"
            )

        else:

            for _, row in significant_interface.iterrows():

                f.write(
                    f"{row['metric_label']}: "
                    f"{row['predictor_1']} vs "
                    f"{row['predictor_2']}; "
                    f"FDR p={row['p_value_fdr']:.6g}; "
                    f"effect="
                    f"{row['rank_biserial_correlation']:.3f}; "
                    f"{row['effect_interpretation']}\n"
                )

        # ----------------------------------------------------
        # Success rate
        # ----------------------------------------------------

        f.write(
            "\nSUCCESS-RATE COMPARISONS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in success_table.iterrows():

            f.write(
                f"{row['predictor_1']} vs "
                f"{row['predictor_2']}: "
                f"{row['success_rate_1_percent']:.2f}% vs "
                f"{row['success_rate_2_percent']:.2f}%; "
                f"FDR p="
                f"{row['p_value_fdr']:.6g}\n"
            )

        # ----------------------------------------------------
        # Primary category Friedman
        # ----------------------------------------------------

        f.write(
            "\nCATEGORY-SPECIFIC FRIEDMAN TESTS ? "
            "PRIMARY METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in category_friedman.iterrows():

            f.write(
                f"{row['category']} / "
                f"{row['metric']}: "
                f"n={int(row['n_complete_structures'])}, "
                f"chi2={row['friedman_chi_square']:.4f}, "
                f"p={row['p_value']:.6g}\n"
            )

        # ----------------------------------------------------
        # Interface category Friedman
        # ----------------------------------------------------

        f.write(
            "\nCATEGORY-SPECIFIC FRIEDMAN TESTS ? "
            "INTERFACE / QUATERNARY METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in category_interface_friedman.iterrows():

            f.write(
                f"{row['category']} / "
                f"{row['metric_label']}: "
                f"n={int(row['n_complete_structures'])}, "
                f"chi2={row['friedman_chi_square']:.4f}, "
                f"p={row['p_value']:.6g}\n"
            )

        # ----------------------------------------------------
        # Significant interface category pairwise
        # ----------------------------------------------------

        f.write(
            "\nSIGNIFICANT CATEGORY-SPECIFIC "
            "INTERFACE / QUATERNARY PAIRWISE COMPARISONS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        significant_category_interface = (
            category_interface_pairwise[
                category_interface_pairwise[
                    "significant_fdr"
                ]
            ]
        )

        if significant_category_interface.empty:

            f.write(
                "No significant category-specific pairwise "
                "comparisons after FDR correction.\n"
            )

        else:

            for _, row in (
                significant_category_interface.iterrows()
            ):

                f.write(
                    f"{row['category']} / "
                    f"{row['metric_label']}: "
                    f"{row['predictor_1']} vs "
                    f"{row['predictor_2']}; "
                    f"FDR p={row['p_value_fdr']:.6g}; "
                    f"effect="
                    f"{row['rank_biserial_correlation']:.3f}; "
                    f"{row['effect_interpretation']}\n"
                )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("03 STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nTables saved to:\n"
        f"  {table_dir}/"
    )

    print(
        f"\nFigures saved to:\n"
        f"  {figure_dir}/"
    )

    print(
        f"\nStatistical report saved to:\n"
        f"  {stats_file}"
    )

    print("\nMain statistical tables:")
    print("  friedman_tests.csv")
    print("  pairwise_wilcoxon.csv")
    print("  effect_sizes.csv")
    print("  interface_friedman_tests.csv")
    print("  interface_pairwise_wilcoxon.csv")
    print("  interface_effect_sizes.csv")
    print("  success_rate_mcnemar.csv")
    print("  runtime_friedman.csv")
    print("  runtime_pairwise_wilcoxon.csv")
    print("  category_friedman.csv")
    print("  category_pairwise_wilcoxon.csv")
    print("  category_interface_friedman.csv")
    print("  category_interface_pairwise_wilcoxon.csv")
    print("  category_interface_effect_sizes.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()