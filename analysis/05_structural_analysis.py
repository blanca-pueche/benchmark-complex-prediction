#!/usr/bin/env python3

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# PREDICTORS AND VARIABLES
# ============================================================

PREDICTORS = [
    "AF3",
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold"
]

QUALITY_METRICS = [
    "num_clashes",
    "num_bad_bonds",
    "num_bad_angles"
]

QUALITY_LABELS = {
    "num_clashes": "Clashes",
    "num_bad_bonds": "Bad bonds",
    "num_bad_angles": "Bad angles"
}

CATEGORIES = [
    "Protein-Protein",
    "Protein-Antibody",
    "Protein-DNA",
    "Protein-RNA",
    "Protein-Peptide"
]

NORMALIZATION_FACTOR = 100


# ============================================================
# HELPERS
# ============================================================

def save_text_report(path, text):
    """
    Save a text report.
    """
    with open(path, "w") as f:
        f.write(text)


def fdr_correct(pvalues):
    """
    Benjamini-Hochberg FDR correction.

    NaN values are preserved.
    """
    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    adjusted = np.full(
        len(pvalues),
        np.nan
    )

    if len(pvalues) == 0:
        return adjusted

    valid = np.isfinite(pvalues)

    if valid.sum() > 0:
        adjusted[valid] = multipletests(
            pvalues[valid],
            method="fdr_bh"
        )[1]

    return adjusted


def get_successful_data(df):
    """
    Return only successful predictions.
    """
    if "prediction_success" in df.columns:

        return df[
            df["prediction_success"] == 1
        ].copy()

    if "status" in df.columns:

        return df[
            df["status"]
            .astype(str)
            .str.upper()
            .eq("SUCCESS")
        ].copy()

    if "lddt" in df.columns:

        return df[
            df["lddt"].notna()
        ].copy()

    raise ValueError(
        "Could not determine prediction success."
    )


def rank_biserial_correlation(x, y):
    """
    Calculate paired rank-biserial correlation.

    Positive values indicate larger values for x than y.
    Negative values indicate larger values for y than x.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    differences = x - y

    nonzero = differences != 0

    differences = differences[nonzero]

    if len(differences) == 0:
        return 0.0

    ranks = pd.Series(
        np.abs(differences)
    ).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = ranks[
        differences > 0
    ].sum()

    negative_rank_sum = ranks[
        differences < 0
    ].sum()

    total_rank_sum = (
        positive_rank_sum
        +
        negative_rank_sum
    )

    if total_rank_sum == 0:
        return 0.0

    return (
        positive_rank_sum
        -
        negative_rank_sum
    ) / total_rank_sum


def interpret_effect(effect):
    """
    Qualitative interpretation of rank-biserial correlation.
    """

    absolute = abs(effect)

    if absolute < 0.1:
        return "negligible"

    elif absolute < 0.3:
        return "small"

    elif absolute < 0.5:
        return "moderate"

    else:
        return "large"


def paired_friedman(
    df,
    metric,
    predictors
):
    """
    Friedman test using complete paired structures.
    """

    pivot = df.pivot_table(
        index="pdb_id",
        columns="predictor",
        values=metric,
        aggfunc="first"
    )

    required = [
        predictor
        for predictor in predictors
        if predictor in pivot.columns
    ]

    if len(required) < 3:
        return None

    paired = pivot[
        required
    ].dropna()

    if len(paired) < 3:
        return None

    arrays = [
        paired[predictor].values
        for predictor in required
    ]

    try:

        statistic, pvalue = friedmanchisquare(
            *arrays
        )

    except Exception:
        return None

    return {
        "chi2": statistic,
        "pvalue": pvalue,
        "n": len(paired),
        "predictors": required
    }


def paired_wilcoxon(
    df,
    metric,
    predictor_x,
    predictor_y
):
    """
    Paired Wilcoxon signed-rank test.
    """

    pivot = df.pivot_table(
        index="pdb_id",
        columns="predictor",
        values=metric,
        aggfunc="first"
    )

    if (
        predictor_x not in pivot.columns
        or
        predictor_y not in pivot.columns
    ):
        return None

    paired = pivot[
        [
            predictor_x,
            predictor_y
        ]
    ].dropna()

    if len(paired) < 3:
        return None

    x = paired[
        predictor_x
    ].values

    y = paired[
        predictor_y
    ].values

    differences = x - y

    if np.allclose(
        differences,
        0
    ):
        statistic = 0.0
        pvalue = 1.0

    else:

        try:

            statistic, pvalue = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

        except Exception:

            return None

    effect = rank_biserial_correlation(
        x,
        y
    )

    median_difference = np.median(
        differences
    )

    return {
        "n": len(paired),
        "statistic": statistic,
        "pvalue": pvalue,
        "rank_biserial": effect,
        "effect_interpretation": interpret_effect(
            effect
        ),
        "median_difference": median_difference
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Structural quality analysis of biomolecular "
            "complex structure predictions."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input benchmark_dataset.csv file."
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=(
            "Output directory for structural-quality "
            "tables, figures, and report."
        )
    )

    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir

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
    print("05 - STRUCTURAL QUALITY ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset:")
    print(input_file)

    df = pd.read_csv(
        input_file
    )

    print(
        f"Prediction rows: {len(df)}"
    )

    if "pdb_id" in df.columns:

        print(
            f"Structures: "
            f"{df['pdb_id'].nunique()}"
        )

    if "predictor" in df.columns:

        print(
            f"Predictors: "
            f"{df['predictor'].nunique()}"
        )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "pdb_id",
        "predictor",
        "category",
        "structure_residues"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            +
            ", ".join(missing)
        )


    missing_quality = [
        metric
        for metric in QUALITY_METRICS
        if metric not in df.columns
    ]

    if missing_quality:

        raise ValueError(
            "Missing structural-quality columns: "
            +
            ", ".join(missing_quality)
        )


    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    numeric_columns = [
        "structure_residues"
    ] + QUALITY_METRICS

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # ========================================================
    # PREDICTION SUCCESS
    # ========================================================

    if "prediction_success" not in df.columns:

        if "status" in df.columns:

            df["prediction_success"] = (
                df["status"]
                .astype(str)
                .str.upper()
                .eq("SUCCESS")
                .astype(int)
            )

        elif "lddt" in df.columns:

            df["prediction_success"] = (
                df["lddt"]
                .notna()
                .astype(int)
            )

        else:

            raise ValueError(
                "Could not determine prediction success."
            )

    else:

        if df["prediction_success"].dtype == bool:

            df["prediction_success"] = (
                df["prediction_success"]
                .astype(int)
            )

        else:

            success_map = {
                "TRUE": 1,
                "FALSE": 0,
                "True": 1,
                "False": 0,
                "SUCCESS": 1,
                "FAIL": 0,
                "FAILED": 0,
                "1": 1,
                "0": 0
            }

            converted = (
                df["prediction_success"]
                .astype(str)
                .map(success_map)
            )

            numeric_original = pd.to_numeric(
                df["prediction_success"],
                errors="coerce"
            )

            df["prediction_success"] = (
                converted
                .fillna(numeric_original)
            )

            df["prediction_success"] = (
                df["prediction_success"]
                .fillna(0)
                .astype(int)
            )


    # ========================================================
    # SUCCESSFUL PREDICTIONS
    # ========================================================

    successful_df = get_successful_data(
        df
    )

    print(
        f"\nSuccessful prediction rows: "
        f"{len(successful_df)}"
    )


    # ========================================================
    # NORMALIZED QUALITY METRICS
    # ========================================================

    for metric in QUALITY_METRICS:

        normalized_name = (
            metric
            + "_per_100_residues"
        )

        successful_df[
            normalized_name
        ] = (
            successful_df[metric]
            /
            successful_df["structure_residues"]
            *
            NORMALIZATION_FACTOR
        )


    NORMALIZED_METRICS = [
        metric
        + "_per_100_residues"
        for metric in QUALITY_METRICS
    ]


    # ========================================================
    # 1. DESCRIPTIVE STRUCTURAL QUALITY
    # ========================================================

    print("\n" + "=" * 70)
    print("1. DESCRIPTIVE STRUCTURAL QUALITY")
    print("=" * 70)

    descriptive_results = []

    for predictor in PREDICTORS:

        subset = successful_df[
            successful_df["predictor"] == predictor
        ]

        for metric in QUALITY_METRICS:

            values = subset[
                metric
            ].dropna()

            if len(values) == 0:
                continue

            normalized = subset[
                metric
                + "_per_100_residues"
            ].dropna()

            descriptive_results.append({
                "predictor": predictor,
                "metric": metric,
                "n": len(values),

                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),

                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),

                "min": values.min(),
                "max": values.max(),

                "zero_count": (
                    values == 0
                ).sum(),

                "zero_percent": (
                    values.eq(0).mean()
                    * 100
                ),

                "normalized_mean_per_100": (
                    normalized.mean()
                ),

                "normalized_median_per_100": (
                    normalized.median()
                ),

                "normalized_q25_per_100": (
                    normalized.quantile(0.25)
                ),

                "normalized_q75_per_100": (
                    normalized.quantile(0.75)
                )
            })


    descriptive_df = pd.DataFrame(
        descriptive_results
    )

    descriptive_df.to_csv(
        os.path.join(
            table_dir,
            "structural_quality_summary.csv"
        ),
        index=False
    )

    print(
        "Saved structural_quality_summary.csv"
    )


    # ========================================================
    # 2. ZERO-DEFECT RATES
    # ========================================================

    print("\n" + "=" * 70)
    print("2. ZERO-DEFECT RATES")
    print("=" * 70)

    zero_results = []

    for predictor in PREDICTORS:

        subset = successful_df[
            successful_df["predictor"] == predictor
        ]

        for metric in QUALITY_METRICS:

            values = subset[
                metric
            ].dropna()

            if len(values) == 0:
                continue

            zero_results.append({
                "predictor": predictor,
                "metric": metric,
                "n": len(values),
                "n_zero": int(
                    (values == 0).sum()
                ),
                "zero_percent": (
                    (values == 0).mean()
                    * 100
                )
            })


    zero_df = pd.DataFrame(
        zero_results
    )

    zero_df.to_csv(
        os.path.join(
            table_dir,
            "zero_defect_rates.csv"
        ),
        index=False
    )

    print(
        "Saved zero_defect_rates.csv"
    )


    # ========================================================
    # 3. OVERALL FRIEDMAN TESTS
    # ========================================================

    print("\n" + "=" * 70)
    print("3. OVERALL FRIEDMAN TESTS")
    print("=" * 70)

    friedman_results = []

    for metric in QUALITY_METRICS:

        result = paired_friedman(
            successful_df,
            metric,
            PREDICTORS
        )

        if result is None:
            continue

        friedman_results.append({
            "metric": metric,
            "chi2": result["chi2"],
            "pvalue": result["pvalue"],
            "n": result["n"]
        })


    friedman_df = pd.DataFrame(
        friedman_results
    )

    if not friedman_df.empty:

        friedman_df[
            "pvalue_fdr"
        ] = fdr_correct(
            friedman_df["pvalue"]
        )


    friedman_df.to_csv(
        os.path.join(
            table_dir,
            "friedman_structural_quality.csv"
        ),
        index=False
    )

    print(
        "Saved friedman_structural_quality.csv"
    )


    # ========================================================
    # 4. OVERALL PAIRWISE WILCOXON TESTS
    # ========================================================

    print("\n" + "=" * 70)
    print("4. OVERALL PAIRWISE WILCOXON TESTS")
    print("=" * 70)

    pairwise_results = []

    for metric in QUALITY_METRICS:

        for i in range(
            len(PREDICTORS)
        ):

            for j in range(
                i + 1,
                len(PREDICTORS)
            ):

                predictor_x = PREDICTORS[i]
                predictor_y = PREDICTORS[j]

                result = paired_wilcoxon(
                    successful_df,
                    metric,
                    predictor_x,
                    predictor_y
                )

                if result is None:
                    continue

                pairwise_results.append({
                    "metric": metric,
                    "predictor_x": predictor_x,
                    "predictor_y": predictor_y,
                    "n": result["n"],
                    "wilcoxon_statistic": result[
                        "statistic"
                    ],
                    "pvalue": result["pvalue"],
                    "median_difference_x_minus_y":
                        result[
                            "median_difference"
                        ],
                    "rank_biserial":
                        result[
                            "rank_biserial"
                        ],
                    "effect_interpretation":
                        result[
                            "effect_interpretation"
                        ]
                })


    pairwise_df = pd.DataFrame(
        pairwise_results
    )

    # FDR separately for each structural-quality metric
    # across the 10 predictor-pair comparisons.

    if not pairwise_df.empty:

        pairwise_df[
            "pvalue_fdr"
        ] = np.nan

        for metric in QUALITY_METRICS:

            mask = (
                pairwise_df["metric"]
                == metric
            )

            pvalues = pairwise_df.loc[
                mask,
                "pvalue"
            ].values

            if len(pvalues) == 0:
                continue

            pairwise_df.loc[
                mask,
                "pvalue_fdr"
            ] = multipletests(
                pvalues,
                method="fdr_bh"
            )[1]


    pairwise_df.to_csv(
        os.path.join(
            table_dir,
            "pairwise_wilcoxon_structural_quality.csv"
        ),
        index=False
    )

    print(
        "Saved pairwise_wilcoxon_structural_quality.csv"
    )


    # ========================================================
    # 5. NORMALIZED STRUCTURAL QUALITY
    # ========================================================

    print("\n" + "=" * 70)
    print("5. NORMALIZED STRUCTURAL QUALITY")
    print("=" * 70)

    normalized_friedman_results = []

    for metric in NORMALIZED_METRICS:

        result = paired_friedman(
            successful_df,
            metric,
            PREDICTORS
        )

        if result is None:
            continue

        normalized_friedman_results.append({
            "metric": metric,
            "chi2": result["chi2"],
            "pvalue": result["pvalue"],
            "n": result["n"]
        })


    normalized_friedman_df = pd.DataFrame(
        normalized_friedman_results
    )

    if not normalized_friedman_df.empty:

        normalized_friedman_df[
            "pvalue_fdr"
        ] = fdr_correct(
            normalized_friedman_df["pvalue"]
        )


    normalized_friedman_df.to_csv(
        os.path.join(
            table_dir,
            "friedman_normalized_quality.csv"
        ),
        index=False
    )


    normalized_pairwise_results = []

    for metric in NORMALIZED_METRICS:

        for i in range(
            len(PREDICTORS)
        ):

            for j in range(
                i + 1,
                len(PREDICTORS)
            ):

                predictor_x = PREDICTORS[i]
                predictor_y = PREDICTORS[j]

                result = paired_wilcoxon(
                    successful_df,
                    metric,
                    predictor_x,
                    predictor_y
                )

                if result is None:
                    continue

                normalized_pairwise_results.append({
                    "metric": metric,
                    "predictor_x": predictor_x,
                    "predictor_y": predictor_y,
                    "n": result["n"],
                    "wilcoxon_statistic":
                        result["statistic"],
                    "pvalue": result["pvalue"],
                    "median_difference_x_minus_y":
                        result["median_difference"],
                    "rank_biserial":
                        result["rank_biserial"],
                    "effect_interpretation":
                        result["effect_interpretation"]
                })


    normalized_pairwise_df = pd.DataFrame(
        normalized_pairwise_results
    )

    if not normalized_pairwise_df.empty:

        normalized_pairwise_df[
            "pvalue_fdr"
        ] = np.nan

        for metric in NORMALIZED_METRICS:

            mask = (
                normalized_pairwise_df["metric"]
                == metric
            )

            pvalues = (
                normalized_pairwise_df
                .loc[mask, "pvalue"]
                .values
            )

            if len(pvalues) == 0:
                continue

            normalized_pairwise_df.loc[
                mask,
                "pvalue_fdr"
            ] = multipletests(
                pvalues,
                method="fdr_bh"
            )[1]


    normalized_pairwise_df.to_csv(
        os.path.join(
            table_dir,
            "pairwise_wilcoxon_normalized_quality.csv"
        ),
        index=False
    )

    print(
        "Saved normalized structural-quality statistics"
    )


    # ========================================================
    # 6. STRUCTURAL QUALITY BY COMPLEX CATEGORY
    # ========================================================

    print("\n" + "=" * 70)
    print("6. STRUCTURAL QUALITY BY COMPLEX CATEGORY")
    print("=" * 70)


    category_descriptive_results = []

    for category in CATEGORIES:

        category_df = successful_df[
            successful_df["category"]
            == category
        ]

        for predictor in PREDICTORS:

            subset = category_df[
                category_df["predictor"]
                == predictor
            ]

            for metric in QUALITY_METRICS:

                values = subset[
                    metric
                ].dropna()

                if len(values) == 0:
                    continue

                normalized = subset[
                    metric
                    + "_per_100_residues"
                ].dropna()

                category_descriptive_results.append({
                    "category": category,
                    "predictor": predictor,
                    "metric": metric,
                    "n": len(values),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "mean": values.mean(),
                    "zero_percent": (
                        values.eq(0).mean()
                        * 100
                    ),
                    "normalized_median_per_100":
                        normalized.median()
                })


    category_descriptive_df = pd.DataFrame(
        category_descriptive_results
    )

    category_descriptive_df.to_csv(
        os.path.join(
            table_dir,
            "structural_quality_by_category.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Category-specific Friedman tests
    # --------------------------------------------------------

    category_friedman_results = []

    for category in CATEGORIES:

        category_df = successful_df[
            successful_df["category"]
            == category
        ]

        for metric in NORMALIZED_METRICS:

            result = paired_friedman(
                category_df,
                metric,
                PREDICTORS
            )

            if result is None:
                continue

            category_friedman_results.append({
                "category": category,
                "metric": metric,
                "chi2": result["chi2"],
                "pvalue": result["pvalue"],
                "n": result["n"]
            })


    category_friedman_df = pd.DataFrame(
        category_friedman_results
    )

    if not category_friedman_df.empty:

        category_friedman_df[
            "pvalue_fdr"
        ] = np.nan

        for category in CATEGORIES:

            mask = (
                category_friedman_df[
                    "category"
                ]
                == category
            )

            pvalues = (
                category_friedman_df
                .loc[mask, "pvalue"]
                .values
            )

            if len(pvalues) == 0:
                continue

            category_friedman_df.loc[
                mask,
                "pvalue_fdr"
            ] = multipletests(
                pvalues,
                method="fdr_bh"
            )[1]


    category_friedman_df.to_csv(
        os.path.join(
            table_dir,
            "category_friedman_normalized_quality.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Category-specific pairwise Wilcoxon
    # --------------------------------------------------------

    category_pairwise_results = []

    for category in CATEGORIES:

        category_df = successful_df[
            successful_df["category"]
            == category
        ]

        for metric in NORMALIZED_METRICS:

            for i in range(
                len(PREDICTORS)
            ):

                for j in range(
                    i + 1,
                    len(PREDICTORS)
                ):

                    predictor_x = PREDICTORS[i]
                    predictor_y = PREDICTORS[j]

                    result = paired_wilcoxon(
                        category_df,
                        metric,
                        predictor_x,
                        predictor_y
                    )

                    if result is None:
                        continue

                    category_pairwise_results.append({
                        "category": category,
                        "metric": metric,
                        "predictor_x":
                            predictor_x,
                        "predictor_y":
                            predictor_y,
                        "n": result["n"],
                        "wilcoxon_statistic":
                            result["statistic"],
                        "pvalue":
                            result["pvalue"],
                        "median_difference_x_minus_y":
                            result[
                                "median_difference"
                            ],
                        "rank_biserial":
                            result[
                                "rank_biserial"
                            ],
                        "effect_interpretation":
                            result[
                                "effect_interpretation"
                            ]
                    })


    category_pairwise_df = pd.DataFrame(
        category_pairwise_results
    )

    if not category_pairwise_df.empty:

        category_pairwise_df[
            "pvalue_fdr"
        ] = np.nan

        # FDR correction separately within
        # category - metric.

        for category in CATEGORIES:

            for metric in NORMALIZED_METRICS:

                mask = (
                    (category_pairwise_df[
                        "category"
                    ] == category)
                    &
                    (category_pairwise_df[
                        "metric"
                    ] == metric)
                )

                pvalues = (
                    category_pairwise_df
                    .loc[mask, "pvalue"]
                    .values
                )

                if len(pvalues) == 0:
                    continue

                category_pairwise_df.loc[
                    mask,
                    "pvalue_fdr"
                ] = multipletests(
                    pvalues,
                    method="fdr_bh"
                )[1]


    category_pairwise_df.to_csv(
        os.path.join(
            table_dir,
            "category_pairwise_normalized_quality.csv"
        ),
        index=False
    )

    print(
        "Saved category-specific structural-quality statistics"
    )


    # ========================================================
    # 7. FIGURE 1 - RAW STRUCTURAL DEFECTS
    # ========================================================

    print("\n" + "=" * 70)
    print("7. FIGURE 1 - RAW STRUCTURAL DEFECTS")
    print("=" * 70)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.20,
        top=0.88,
        wspace=0.28
    )

    for ax, metric in zip(
        axes,
        QUALITY_METRICS
    ):

        values_by_predictor = []

        labels = []

        for predictor in PREDICTORS:

            values = successful_df[
                successful_df["predictor"]
                == predictor
            ][metric].dropna()

            values_by_predictor.append(
                values
            )

            labels.append(
                predictor
            )

        bp = ax.boxplot(
            values_by_predictor,
            labels=labels,
            showfliers=False,
            patch_artist=True
        )

        for box in bp["boxes"]:
            box.set_alpha(0.75)

        ax.set_title(
            QUALITY_LABELS[metric],
            loc="left",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Predictor",
            fontsize=9
        )

        ax.set_ylabel(
            "Number of defects",
            fontsize=9
        )

        ax.tick_params(
            axis="x",
            labelrotation=35,
            labelsize=8
        )

        ax.tick_params(
            axis="y",
            labelsize=8
        )

        ax.grid(
            axis="y",
            alpha=0.2,
            linewidth=0.6
        )

        for spine in ax.spines.values():
            spine.set_visible(False)


    png_file = os.path.join(
        figure_dir,
        "01_structural_quality_raw.png"
    )

    pdf_file = os.path.join(
        figure_dir,
        "01_structural_quality_raw.pdf"
    )

    fig.savefig(
        png_file,
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved:\n{png_file}"
    )

    print(
        f"Saved:\n{pdf_file}"
    )


    # ========================================================
    # 8. FIGURE 2 - NORMALIZED STRUCTURAL DEFECTS
    # ========================================================

    print("\n" + "=" * 70)
    print("8. FIGURE 2 - NORMALIZED STRUCTURAL DEFECTS")
    print("=" * 70)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.20,
        top=0.88,
        wspace=0.28
    )

    for ax, metric in zip(
        axes,
        NORMALIZED_METRICS
    ):

        base_metric = metric.replace(
            "_per_100_residues",
            ""
        )

        values_by_predictor = []

        labels = []

        for predictor in PREDICTORS:

            values = successful_df[
                successful_df["predictor"]
                == predictor
            ][metric].dropna()

            values_by_predictor.append(
                values
            )

            labels.append(
                predictor
            )

        bp = ax.boxplot(
            values_by_predictor,
            labels=labels,
            showfliers=False,
            patch_artist=True
        )

        for box in bp["boxes"]:
            box.set_alpha(0.75)

        ax.set_title(
            QUALITY_LABELS[base_metric],
            loc="left",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Predictor",
            fontsize=9
        )

        ax.set_ylabel(
            "Defects per 100 residues",
            fontsize=9
        )

        ax.tick_params(
            axis="x",
            labelrotation=35,
            labelsize=8
        )

        ax.tick_params(
            axis="y",
            labelsize=8
        )

        ax.grid(
            axis="y",
            alpha=0.2,
            linewidth=0.6
        )

        for spine in ax.spines.values():
            spine.set_visible(False)


    png_file = os.path.join(
        figure_dir,
        "02_structural_quality_normalized.png"
    )

    pdf_file = os.path.join(
        figure_dir,
        "02_structural_quality_normalized.pdf"
    )

    fig.savefig(
        png_file,
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved:\n{png_file}"
    )

    print(
        f"Saved:\n{pdf_file}"
    )


    # ========================================================
    # 9. FIGURE 3 - ZERO-DEFECT PERCENTAGES
    # ========================================================

    print("\n" + "=" * 70)
    print("9. FIGURE 3 - ZERO-DEFECT PERCENTAGES")
    print("=" * 70)

    zero_pivot = zero_df.pivot(
        index="predictor",
        columns="metric",
        values="zero_percent"
    )

    zero_pivot = zero_pivot.reindex(
        PREDICTORS
    )

    zero_pivot = zero_pivot.rename(
        columns=QUALITY_LABELS
    )

    ax = zero_pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel(
        "Predictor"
    )

    ax.set_ylabel(
        "Predictions with zero defects (%)"
    )

    ax.set_title(
        "Predictions with zero structural defects"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.legend(
        title="Structural defect"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figure_dir,
            "03_zero_defect_percentage.png"
        ),
        dpi=400,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            figure_dir,
            "03_zero_defect_percentage.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 10. SUMMARY REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "05 - STRUCTURAL QUALITY ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        f"Prediction rows: {len(df)}"
    )

    report_lines.append(
        f"Successful prediction rows: "
        f"{len(successful_df)}"
    )

    report_lines.append(
        f"Structures: {df['pdb_id'].nunique()}"
    )

    report_lines.append(
        f"Predictors: {df['predictor'].nunique()}"
    )

    report_lines.append("")

    report_lines.append(
        "STRUCTURAL QUALITY METRICS"
    )

    report_lines.append(
        "1. Number of steric clashes"
    )

    report_lines.append(
        "2. Number of bad bonds"
    )

    report_lines.append(
        "3. Number of bad angles"
    )

    report_lines.append("")

    report_lines.append(
        "ANALYSES PERFORMED"
    )

    report_lines.append(
        "1. Descriptive statistics"
    )

    report_lines.append(
        "2. Zero-defect percentages"
    )

    report_lines.append(
        "3. Overall Friedman tests"
    )

    report_lines.append(
        "4. Overall paired Wilcoxon tests"
    )

    report_lines.append(
        "5. Size-normalized defects per 100 residues"
    )

    report_lines.append(
        "6. Category-specific structural-quality analyses"
    )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        "Lower numbers of clashes, bad bonds, and bad angles "
        "indicate fewer structural defects."
    )

    report_lines.append(
        "Raw defect counts are affected by structure size."
    )

    report_lines.append(
        "Normalized analyses therefore report defects per 100 residues."
    )

    report_lines.append(
        "Failed predictions are excluded from structural-quality "
        "analyses and are not assigned zero defects."
    )

    report_lines.append(
        "Prediction success is analyzed separately from structural quality."
    )

    report_path = os.path.join(
        output_dir,
        "structural_quality_results.txt"
    )

    save_text_report(
        report_path,
        "\n".join(report_lines)
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("STRUCTURAL QUALITY ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nTables saved to:")
    print(table_dir)

    print("\nFigures saved to:")
    print(figure_dir)

    print("\nSummary saved to:")
    print(report_path)

    print("\nDone.")


if __name__ == "__main__":
    main()