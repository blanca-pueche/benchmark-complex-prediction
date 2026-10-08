#!/usr/bin/env python3

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTORS = [
    "AF3",
    "Boltz",
    "Chai",
    "Protenix",
    "IntelliFold"
]

PRIMARY_METRICS = [
    "lddt",
    "tm_score",
    "rmsd",
    "dockq_ave"
]

STRUCTURAL_VARIABLES = [
    "structure_residues",
    "structure_chain_count",
    "resolution"
]

CATEGORIES = [
    "Protein-Protein",
    "Protein-Antibody",
    "Protein-DNA",
    "Protein-RNA",
    "Protein-Peptide",
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find the first column present in the dataframe from a list
    of possible names.
    """
    for column in candidates:
        if column in df.columns:
            return column

    return None


def fdr_correct(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvalues = np.asarray(pvalues, dtype=float)

    if len(pvalues) == 0:
        return np.array([])

    valid = np.isfinite(pvalues)

    adjusted = np.full(len(pvalues), np.nan)

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
        return df[df["prediction_success"] == 1].copy()

    if "status" in df.columns:
        return df[
            df["status"].astype(str).str.upper() == "SUCCESS"
        ].copy()

    return df[df["lddt"].notna()].copy()


def safe_spearman(x, y, min_n=3):
    """
    Safely calculate Spearman correlation.

    Returns:
        rho, pvalue, n

    Returns NaN when the correlation cannot be calculated.
    """

    valid = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    n = len(valid)

    if n < min_n:
        return np.nan, np.nan, n

    # Correlation is undefined if either variable is constant
    if valid["x"].nunique() < 2:
        return np.nan, np.nan, n

    if valid["y"].nunique() < 2:
        return np.nan, np.nan, n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        rho, pvalue = spearmanr(
            valid["x"],
            valid["y"]
        )

    return rho, pvalue, n


def save_text_report(path, text):
    """
    Save a text report.
    """
    with open(path, "w") as f:
        f.write(text)


def main():

    parser = argparse.ArgumentParser(
        description="Performance determinants analysis of the biomolecular structure prediction benchmark."
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
        help="Output directory for tables, figures, and the summary report."
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


    # ============================================================
    # LOAD DATA
    # ============================================================

    print("=" * 70)
    print("04 - PERFORMANCE DETERMINANTS ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset:")
    print(input_file)

    df = pd.read_csv(input_file)

    print(f"Dataset: {len(df)} prediction rows")

    if "pdb_id" in df.columns:
        print(f"Structures: {df['pdb_id'].nunique()}")

    if "predictor" in df.columns:
        print(f"Predictors: {df['predictor'].nunique()}")


    # ============================================================
    # STANDARDIZE DATA
    # ============================================================

    # Convert numeric columns
    numeric_columns = (
        PRIMARY_METRICS
        + STRUCTURAL_VARIABLES
    )

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    # ------------------------------------------------------------
    # Prediction success
    # ------------------------------------------------------------

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
                df["lddt"].notna()
                .astype(int)
            )

        else:

            raise ValueError(
                "Could not determine prediction success."
            )

    else:

        # Convert boolean/object success values to numeric 0/1.
        if df["prediction_success"].dtype == bool:

            df["prediction_success"] = (
                df["prediction_success"]
                .astype(int)
            )

        else:

            # Handle possible string values
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
                .astype(float)
            )


    df["prediction_success"] = (
        df["prediction_success"]
        .fillna(0)
        .astype(int)
    )


    # ============================================================
    # STRUCTURE-LEVEL DATASET
    # ============================================================

    structure_columns = [
        "pdb_id",
        "category",
        "structure_residues",
        "structure_chain_count",
        "resolution"
    ]

    structure_columns = [
        column
        for column in structure_columns
        if column in df.columns
    ]

    structure_df = (
        df[structure_columns]
        .drop_duplicates(subset=["pdb_id"])
        .copy()
    )

    print(
        f"Structure-level dataset: "
        f"{len(structure_df)} structures"
    )


    # ============================================================
    # STRUCTURE SIZE GROUPS
    # ============================================================

    if "structure_residues" in structure_df.columns:

        valid_sizes = structure_df[
            "structure_residues"
        ].dropna()

        if len(valid_sizes) >= 3:

            q1 = valid_sizes.quantile(1 / 3)
            q2 = valid_sizes.quantile(2 / 3)

            def size_group(value):

                if pd.isna(value):
                    return np.nan

                if value <= q1:
                    return "Small"

                elif value <= q2:
                    return "Medium"

                else:
                    return "Large"

            structure_df["size_group"] = (
                structure_df["structure_residues"]
                .apply(size_group)
            )

        else:

            structure_df["size_group"] = np.nan


    # ============================================================
    # CHAIN COUNT GROUPS
    # ============================================================

    if "structure_chain_count" in structure_df.columns:

        # Treat chain count as a discrete variable.
        # Each observed number of chains is kept as its own group.
        structure_df["chain_group"] = (
            structure_df["structure_chain_count"]
            .round()
            .astype("Int64")
            .astype(str)
        )


    # Merge structural groups back into prediction-level data

    group_columns = [
        "pdb_id",
        "size_group",
        "chain_group"
    ]

    group_columns = [
        column
        for column in group_columns
        if column in structure_df.columns
    ]

    df = df.merge(
        structure_df[group_columns],
        on="pdb_id",
        how="left"
    )


    # ============================================================
    # 1. SPEARMAN CORRELATIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("1. SPEARMAN CORRELATIONS")
    print("=" * 70)

    correlation_results = []

    successful_df = get_successful_data(df)

    for predictor in PREDICTORS:

        predictor_df = successful_df[
            successful_df["predictor"] == predictor
        ].copy()

        for metric in PRIMARY_METRICS:

            if metric not in predictor_df.columns:
                continue

            for variable in STRUCTURAL_VARIABLES:

                if variable not in predictor_df.columns:
                    continue

                rho, pvalue, n = safe_spearman(
                    predictor_df[variable],
                    predictor_df[metric]
                )

                correlation_results.append({
                    "predictor": predictor,
                    "metric": metric,
                    "structural_variable": variable,
                    "rho": rho,
                    "pvalue": pvalue,
                    "n": n
                })


    correlations_df = pd.DataFrame(
        correlation_results
    )

    if not correlations_df.empty:

        correlations_df["pvalue_fdr"] = fdr_correct(
            correlations_df["pvalue"]
        )

    correlations_df.to_csv(
        os.path.join(
            table_dir,
            "correlations.csv"
        ),
        index=False
    )

    print("Saved correlations.csv")


    # ============================================================
    # 2. CORRELATIONS BY CATEGORY
    # ============================================================

    print("\n" + "=" * 70)
    print("2. CORRELATIONS BY CATEGORY")
    print("=" * 70)

    category_correlation_results = []

    for predictor in PREDICTORS:

        for category in CATEGORIES:

            subset = successful_df[
                (successful_df["predictor"] == predictor)
                &
                (successful_df["category"] == category)
            ].copy()

            for metric in PRIMARY_METRICS:

                if metric not in subset.columns:
                    continue

                for variable in STRUCTURAL_VARIABLES:

                    if variable not in subset.columns:
                        continue

                    rho, pvalue, n = safe_spearman(
                        subset[variable],
                        subset[metric],
                        min_n=3
                    )

                    category_correlation_results.append({
                        "predictor": predictor,
                        "category": category,
                        "metric": metric,
                        "structural_variable": variable,
                        "rho": rho,
                        "pvalue": pvalue,
                        "n": n
                    })


    correlations_by_category = pd.DataFrame(
        category_correlation_results
    )

    if not correlations_by_category.empty:

        correlations_by_category["pvalue_fdr"] = fdr_correct(
            correlations_by_category["pvalue"]
        )

    correlations_by_category.to_csv(
        os.path.join(
            table_dir,
            "correlations_by_category.csv"
        ),
        index=False
    )

    print("Saved correlations_by_category.csv")


    # ============================================================
    # 3. PERFORMANCE BY STRUCTURE SIZE
    # ============================================================

    print("\n" + "=" * 70)
    print("3. PERFORMANCE BY STRUCTURE SIZE")
    print("=" * 70)

    size_results = []

    if "size_group" in successful_df.columns:

        for predictor in PREDICTORS:

            for group in ["Small", "Medium", "Large"]:

                subset = successful_df[
                    (successful_df["predictor"] == predictor)
                    &
                    (successful_df["size_group"] == group)
                ]

                for metric in PRIMARY_METRICS:

                    if metric not in subset.columns:
                        continue

                    values = subset[metric].dropna()

                    if len(values) == 0:
                        continue

                    size_results.append({
                        "predictor": predictor,
                        "size_group": group,
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean(),
                        "median": values.median(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max()
                    })


    performance_by_size = pd.DataFrame(
        size_results
    )

    performance_by_size.to_csv(
        os.path.join(
            table_dir,
            "performance_by_size_group.csv"
        ),
        index=False
    )

    print("Saved performance_by_size_group.csv")


    # ============================================================
    # 4. PERFORMANCE BY CHAIN COUNT
    # ============================================================

    print("\n" + "=" * 70)
    print("4. PERFORMANCE BY CHAIN COUNT")
    print("=" * 70)

    chain_results = []

    if "chain_group" in successful_df.columns:

        chain_groups = sorted(
            successful_df["chain_group"]
            .dropna()
            .unique(),
            key=lambda x: int(x)
        )

        for predictor in PREDICTORS:

            for group in chain_groups:

                subset = successful_df[
                    (successful_df["predictor"] == predictor)
                    &
                    (successful_df["chain_group"] == group)
                ]

                for metric in PRIMARY_METRICS:

                    if metric not in subset.columns:
                        continue

                    values = subset[metric].dropna()

                    if len(values) == 0:
                        continue

                    chain_results.append({
                        "predictor": predictor,
                        "chain_group": group,
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean(),
                        "median": values.median(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max()
                    })


    performance_by_chain = pd.DataFrame(
        chain_results
    )

    performance_by_chain.to_csv(
        os.path.join(
            table_dir,
            "performance_by_chain_group.csv"
        ),
        index=False
    )

    print("Saved performance_by_chain_group.csv")


    # ============================================================
    # 5. MULTIVARIABLE REGRESSION
    # ============================================================

    print("\n" + "=" * 70)
    print("5. MULTIVARIABLE REGRESSION")
    print("=" * 70)

    regression_results = []

    for predictor in PREDICTORS:

        predictor_df = successful_df[
            successful_df["predictor"] == predictor
        ].copy()

        for metric in PRIMARY_METRICS:

            required = [
                metric,
                "structure_residues",
                "structure_chain_count",
                "resolution",
                "category"
            ]

            required = [
                column
                for column in required
                if column in predictor_df.columns
            ]

            if len(required) < 4:
                continue

            model_df = predictor_df[
                required
            ].dropna().copy()

            if len(model_df) < 20:
                continue

            formula = (
                f"{metric} ~ "
                "structure_residues + "
                "structure_chain_count + "
                "resolution + "
                "C(category)"
            )

            try:

                model = smf.ols(
                    formula=formula,
                    data=model_df
                ).fit()

                for term in model.params.index:

                    regression_results.append({
                        "predictor": predictor,
                        "metric": metric,
                        "term": term,
                        "coefficient": model.params[term],
                        "std_error": model.bse[term],
                        "t_value": model.tvalues[term],
                        "pvalue": model.pvalues[term],
                        "r_squared": model.rsquared,
                        "adj_r_squared": model.rsquared_adj,
                        "n": int(model.nobs)
                    })

            except Exception as e:

                print(
                    f"Regression failed for "
                    f"{predictor} / {metric}: {e}"
                )


    regression_df = pd.DataFrame(
        regression_results
    )


    # ------------------------------------------------------------
    # Benjamini-Hochberg FDR correction
    #
    # Correction is performed separately within each
    # predictor x metric regression model and excludes the
    # intercept. This treats the explanatory terms of each
    # regression as one hypothesis-testing family.
    # ------------------------------------------------------------

    if not regression_df.empty:

        regression_df["pvalue_fdr"] = np.nan

        for predictor in PREDICTORS:

            for metric in PRIMARY_METRICS:

                mask = (
                    (regression_df["predictor"] == predictor)
                    &
                    (regression_df["metric"] == metric)
                    &
                    (regression_df["term"] != "Intercept")
                )

                pvalues = (
                    regression_df.loc[
                        mask,
                        "pvalue"
                    ].values
                )

                if len(pvalues) == 0:
                    continue

                regression_df.loc[
                    mask,
                    "pvalue_fdr"
                ] = multipletests(
                    pvalues,
                    alpha=0.05,
                    method="fdr_bh"
                )[1]


    regression_df.to_csv(
        os.path.join(
            table_dir,
            "regression_models.csv"
        ),
        index=False
    )

    print("Saved regression_models.csv")


    # ============================================================
    # 6. FAILURE ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("6. FAILURE ANALYSIS")
    print("=" * 70)

    failure_results = []

    for predictor in PREDICTORS:

        predictor_df = df[
            df["predictor"] == predictor
        ].copy()

        for category in CATEGORIES:

            subset = predictor_df[
                predictor_df["category"] == category
            ]

            n_total = len(subset)

            if n_total == 0:
                continue

            n_success = (
                subset["prediction_success"]
                .sum()
            )

            n_failure = (
                n_total - n_success
            )

            failure_results.append({
                "predictor": predictor,
                "category": category,
                "n_total": n_total,
                "n_success": n_success,
                "n_failure": n_failure,
                "success_rate": n_success / n_total,
                "failure_rate": n_failure / n_total
            })


    failure_rates = pd.DataFrame(
        failure_results
    )

    failure_rates.to_csv(
        os.path.join(
            table_dir,
            "failure_rates_by_category.csv"
        ),
        index=False
    )

    print("Saved failure_rates_by_category.csv")


    # ============================================================
    # 7. FAILURE VS STRUCTURAL COMPLEXITY
    # ============================================================

    print("\n" + "=" * 70)
    print("7. FAILURE VS STRUCTURAL COMPLEXITY")
    print("=" * 70)

    failure_correlation_results = []

    for predictor in PREDICTORS:

        predictor_df = df[
            df["predictor"] == predictor
        ].copy()

        for variable in STRUCTURAL_VARIABLES:

            if variable not in predictor_df.columns:
                continue

            valid = predictor_df[
                [variable, "prediction_success"]
            ].dropna()

            if len(valid) < 5:
                continue

            # Both variables need variation
            if valid[variable].nunique() < 2:
                continue

            if valid["prediction_success"].nunique() < 2:
                continue

            # Failure is coded as 1 here
            failure = (
                1 - valid["prediction_success"]
            )

            rho, pvalue, n = safe_spearman(
                valid[variable],
                failure,
                min_n=5
            )

            failure_correlation_results.append({
                "predictor": predictor,
                "structural_variable": variable,
                "rho": rho,
                "pvalue": pvalue,
                "n": n
            })


    failure_correlations = pd.DataFrame(
        failure_correlation_results
    )

    if not failure_correlations.empty:

        failure_correlations["pvalue_fdr"] = fdr_correct(
            failure_correlations["pvalue"]
        )

    failure_correlations.to_csv(
        os.path.join(
            table_dir,
            "failure_correlations.csv"
        ),
        index=False
    )

    print("Saved failure_correlations.csv")


    # ============================================================
    # 8. LOGISTIC REGRESSION
    # ============================================================

    print("\n" + "=" * 70)
    print("8. LOGISTIC REGRESSION")
    print("=" * 70)

    logistic_results = []

    for predictor in PREDICTORS:

        predictor_df = df[
            df["predictor"] == predictor
        ].copy()

        required = [
            "prediction_success",
            "structure_residues",
            "structure_chain_count",
            "resolution",
            "category"
        ]

        required = [
            column
            for column in required
            if column in predictor_df.columns
        ]

        if len(required) < 5:
            print(
                f"Skipping logistic regression for "
                f"{predictor}: missing variables"
            )
            continue

        model_df = predictor_df[
            required
        ].dropna().copy()

        if len(model_df) < 20:
            print(
                f"Skipping logistic regression for "
                f"{predictor}: insufficient data"
            )
            continue

        # Explicitly convert dependent variable to 0/1
        model_df["prediction_success"] = (
            pd.to_numeric(
                model_df["prediction_success"],
                errors="coerce"
            )
        )

        model_df = model_df.dropna(
            subset=["prediction_success"]
        )

        model_df["prediction_success"] = (
            model_df["prediction_success"]
            .astype(int)
        )

        # Need both successful and failed predictions
        if model_df["prediction_success"].nunique() < 2:

            print(
                f"Skipping logistic regression for "
                f"{predictor}: outcome has only one class"
            )

            continue

        formula = (
            "prediction_success ~ "
            "structure_residues + "
            "structure_chain_count + "
            "resolution + "
            "C(category)"
        )

        try:

            model = smf.logit(
                formula=formula,
                data=model_df
            ).fit(
                disp=False
            )

            for term in model.params.index:

                coefficient = model.params[term]

                logistic_results.append({
                    "predictor": predictor,
                    "term": term,
                    "coefficient": coefficient,
                    "odds_ratio": np.exp(coefficient),
                    "std_error": model.bse[term],
                    "z_value": model.tvalues[term],
                    "pvalue": model.pvalues[term],
                    "pseudo_r_squared": model.prsquared,
                    "n": int(model.nobs)
                })

        except PerfectSeparationError:

            print(
                f"Logistic regression failed for "
                f"{predictor}: perfect separation"
            )

        except np.linalg.LinAlgError as e:

            print(
                f"Logistic regression failed for "
                f"{predictor}: {e}"
            )

        except Exception as e:

            print(
                f"Logistic regression failed for "
                f"{predictor}: {e}"
            )


    logistic_df = pd.DataFrame(
        logistic_results
    )

    if not logistic_df.empty:

        logistic_df["pvalue_fdr"] = fdr_correct(
            logistic_df["pvalue"]
        )

    logistic_df.to_csv(
        os.path.join(
            table_dir,
            "logistic_regression.csv"
        ),
        index=False
    )

    print("Saved logistic_regression.csv")


    # ============================================================
    # 9. FIGURES
    # ============================================================

    print("\n" + "=" * 70)
    print("9. FIGURES")
    print("=" * 70)


    # ============================================================
    # FIGURE 1: PERFORMANCE VS STRUCTURE SIZE
    # ============================================================

    if "structure_residues" in successful_df.columns:

        from statsmodels.nonparametric.smoothers_lowess import lowess

        metric_labels = {
            "lddt": "lDDT",
            "tm_score": "TM-score",
            "rmsd": "RMSD (?)",
            "dockq_ave": "DockQ",
        }

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.10,
            top=0.91,
            wspace=0.28,
            hspace=0.32,
        )

        for panel_label, (ax, metric) in zip(
            ["A", "B", "C", "D"],
            zip(axes.flat, PRIMARY_METRICS)
        ):

            ax.text(
                -0.12,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
                ha="left",
            )

            if metric not in successful_df.columns:
                ax.set_visible(False)
                continue

            for predictor in PREDICTORS:

                subset = successful_df[
                    successful_df["predictor"] == predictor
                ][
                    [
                        "structure_residues",
                        metric
                    ]
                ].dropna()

                if len(subset) < 3:
                    continue

                ax.scatter(
                    subset["structure_residues"],
                    subset[metric],
                    alpha=0.35,
                    s=10,
                    label=predictor,
                )

                try:

                    smoothed = lowess(
                        subset[metric],
                        subset["structure_residues"],
                        frac=0.6,
                        return_sorted=True,
                    )

                    ax.plot(
                        smoothed[:, 0],
                        smoothed[:, 1],
                        linewidth=1.8,
                    )

                except Exception:
                    pass

            ax.set_xlabel(
                "Structure size (residues)",
                fontsize=9,
            )

            ax.set_ylabel(
                metric_labels[metric],
                fontsize=9,
            )

            ax.set_title(
                metric_labels[metric],
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.tick_params(
                labelsize=8,
            )

            ax.grid(
                alpha=0.2,
                linewidth=0.6,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

        handles, labels = axes.flat[0].get_legend_handles_labels()

        if handles:

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.975),
                ncol=5,
                frameon=False,
                fontsize=8.5,
            )

        png_file = os.path.join(
            figure_dir,
            "01_performance_vs_size.png",
        )

        pdf_file = os.path.join(
            figure_dir,
            "01_performance_vs_size.pdf",
        )

        fig.savefig(
            png_file,
            dpi=400,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved:\n{png_file}")
        print(f"Saved:\n{pdf_file}")


    # ============================================================
    # FIGURE 2: PERFORMANCE BY CHAIN COUNT
    # ============================================================

    if "structure_chain_count" in successful_df.columns:

        metric_labels = {
            "lddt": "lDDT",
            "tm_score": "TM-score",
            "rmsd": "RMSD (?)",
            "dockq_ave": "DockQ",
        }

        chain_groups = sorted(
            successful_df["structure_chain_count"]
            .dropna()
            .unique()
        )

        chain_colors = {
            2: "#4C78A8",
            3: "#F58518",
        }

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.15,
            top=0.91,
            wspace=0.28,
            hspace=0.38,
        )

        if len(chain_groups) == 2:
            offsets = [-0.18, 0.18]
        else:
            offsets = np.linspace(
                -0.22,
                0.22,
                len(chain_groups),
            )

        for panel_label, (ax, metric) in zip(
            ["A", "B", "C", "D"],
            zip(axes.flat, PRIMARY_METRICS)
        ):

            ax.text(
                -0.12,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
                ha="left",
            )

            if metric not in successful_df.columns:
                ax.set_visible(False)
                continue

            for i, predictor in enumerate(PREDICTORS):

                center = i + 1

                for j, chain_count in enumerate(chain_groups):

                    values = successful_df[
                        (
                            successful_df["structure_chain_count"]
                            == chain_count
                        )
                        &
                        (
                            successful_df["predictor"]
                            == predictor
                        )
                    ][metric].dropna()

                    if len(values) == 0:
                        continue

                    bp = ax.boxplot(
                        values,
                        positions=[
                            center + offsets[j]
                        ],
                        widths=0.30,
                        patch_artist=True,
                        showfliers=False,
                    )

                    color = chain_colors.get(
                        int(chain_count),
                        None,
                    )

                    if color is not None:

                        for box in bp["boxes"]:
                            box.set_facecolor(color)
                            box.set_alpha(0.75)

                        for element in [
                            "whiskers",
                            "caps",
                            "medians",
                        ]:
                            for line in bp[element]:
                                line.set_color(color)

            ax.set_xticks(
                range(1, len(PREDICTORS) + 1)
            )

            ax.set_xticklabels(
                PREDICTORS,
                fontsize=8,
            )

            ax.set_xlabel(
                "Predictor",
                fontsize=9,
            )

            ax.set_ylabel(
                metric_labels[metric],
                fontsize=9,
            )

            if metric in ["lddt", "tm_score"]:
                ax.set_ylim(0.60, 1.00)
                ax.set_yticks(np.arange(0.60, 1.01, 0.05))

            ax.set_title(
                metric_labels[metric],
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.tick_params(
                axis="y",
                labelsize=8,
            )

            ax.grid(
                axis="y",
                alpha=0.2,
                linewidth=0.6,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=chain_colors[int(chain_count)],
                edgecolor="none",
                alpha=0.75,
                label=f"{int(chain_count)} chains",
            )
            for chain_count in chain_groups
            if int(chain_count) in chain_colors
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=8.5,
        )

        png_file = os.path.join(
            figure_dir,
            "02_performance_vs_chains.png",
        )

        pdf_file = os.path.join(
            figure_dir,
            "02_performance_vs_chains.pdf",
        )

        fig.savefig(
            png_file,
            dpi=400,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved:\n{png_file}")
        print(f"Saved:\n{pdf_file}")


    # ============================================================
    # FIGURE 3: PERFORMANCE BY STRUCTURE SIZE GROUP
    # ============================================================

    if "size_group" in successful_df.columns:

        metric_labels = {
            "lddt": "lDDT",
            "tm_score": "TM-score",
            "rmsd": "RMSD (?)",
            "dockq_ave": "DockQ",
        }

        size_groups = [
            "Small",
            "Medium",
            "Large",
        ]

        size_colors = {
            "Small": "#59A14F",
            "Medium": "#F2CF5B",
            "Large": "#E15759",
        }

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.15,
            top=0.91,
            wspace=0.28,
            hspace=0.38,
        )

        offsets = [-0.24, 0, 0.24]

        for panel_label, (ax, metric) in zip(
            ["A", "B", "C", "D"],
            zip(axes.flat, PRIMARY_METRICS)
        ):

            ax.text(
                -0.12,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
                ha="left",
            )

            if metric not in successful_df.columns:
                ax.set_visible(False)
                continue

            for i, predictor in enumerate(PREDICTORS):

                center = i + 1

                for j, size_group in enumerate(size_groups):

                    values = successful_df[
                        (
                            successful_df["size_group"]
                            == size_group
                        )
                        &
                        (
                            successful_df["predictor"]
                            == predictor
                        )
                    ][metric].dropna()

                    if len(values) == 0:
                        continue

                    bp = ax.boxplot(
                        values,
                        positions=[
                            center + offsets[j]
                        ],
                        widths=0.22,
                        patch_artist=True,
                        showfliers=False,
                    )

                    color = size_colors[size_group]

                    for box in bp["boxes"]:
                        box.set_facecolor(color)
                        box.set_alpha(0.75)

                    for element in [
                        "whiskers",
                        "caps",
                        "medians",
                    ]:
                        for line in bp[element]:
                            line.set_color(color)

            ax.set_xticks(
                range(1, len(PREDICTORS) + 1)
            )

            ax.set_xticklabels(
                PREDICTORS,
                fontsize=8,
            )

            ax.set_xlabel(
                "Predictor",
                fontsize=9,
            )

            ax.set_ylabel(
                metric_labels[metric],
                fontsize=9,
            )

            if metric in ["lddt", "tm_score"]:
                ax.set_ylim(0.4, 1.0)
                ax.set_yticks(np.arange(0.4, 1.01, 0.1))

            ax.set_title(
                metric_labels[metric],
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.tick_params(
                axis="y",
                labelsize=8,
            )

            ax.grid(
                axis="y",
                alpha=0.2,
                linewidth=0.6,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=size_colors[group],
                edgecolor="none",
                alpha=0.75,
                label=group,
            )
            for group in size_groups
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )

        png_file = os.path.join(
            figure_dir,
            "03_performance_by_size_group.png",
        )

        pdf_file = os.path.join(
            figure_dir,
            "03_performance_by_size_group.pdf",
        )

        fig.savefig(
            png_file,
            dpi=400,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved:\n{png_file}")
        print(f"Saved:\n{pdf_file}")


    # ============================================================
    # FIGURE 4: PERFORMANCE VS EXPERIMENTAL RESOLUTION
    # ============================================================

    if "resolution" in successful_df.columns:

        from statsmodels.nonparametric.smoothers_lowess import lowess

        metric_labels = {
            "lddt": "lDDT",
            "tm_score": "TM-score",
            "rmsd": "RMSD (?)",
            "dockq_ave": "DockQ",
        }

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.10,
            top=0.91,
            wspace=0.28,
            hspace=0.32,
        )

        for panel_label, (ax, metric) in zip(
            ["A", "B", "C", "D"],
            zip(axes.flat, PRIMARY_METRICS)
        ):

            ax.text(
                -0.12,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
                ha="left",
            )

            if metric not in successful_df.columns:
                ax.set_visible(False)
                continue

            for predictor in PREDICTORS:

                subset = successful_df[
                    successful_df["predictor"] == predictor
                ][
                    [
                        "resolution",
                        metric
                    ]
                ].dropna()

                if len(subset) < 3:
                    continue

                ax.scatter(
                    subset["resolution"],
                    subset[metric],
                    alpha=0.35,
                    s=10,
                    label=predictor,
                )

                try:

                    smoothed = lowess(
                        subset[metric],
                        subset["resolution"],
                        frac=0.6,
                        return_sorted=True,
                    )

                    ax.plot(
                        smoothed[:, 0],
                        smoothed[:, 1],
                        linewidth=1.8,
                    )

                except Exception:
                    pass

            ax.set_xlabel(
                "Experimental resolution (?)",
                fontsize=9,
            )

            ax.set_ylabel(
                metric_labels[metric],
                fontsize=9,
            )

            ax.set_title(
                metric_labels[metric],
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.tick_params(
                labelsize=8,
            )

            ax.grid(
                alpha=0.2,
                linewidth=0.6,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

        handles, labels = axes.flat[0].get_legend_handles_labels()

        if handles:

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.975),
                ncol=5,
                frameon=False,
                fontsize=8.5,
            )

        png_file = os.path.join(
            figure_dir,
            "04_performance_vs_resolution.png",
        )

        pdf_file = os.path.join(
            figure_dir,
            "04_performance_vs_resolution.pdf",
        )

        fig.savefig(
            png_file,
            dpi=400,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved:\n{png_file}")
        print(f"Saved:\n{pdf_file}")


    # ============================================================
    # FIGURE 5: FAILURE RATE BY CATEGORY
    # ============================================================

    if not failure_rates.empty:

        pivot = failure_rates.pivot(
            index="category",
            columns="predictor",
            values="failure_rate"
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(11, 6)
        )

        ax.set_xlabel(
            "Category"
        )

        ax.set_ylabel(
            "Failure rate"
        )

        ax.set_title(
            "Prediction Failure Rate by Category"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                figure_dir,
                "05_failure_rate_by_category.png"
            ),
            dpi=400,
            bbox_inches="tight",
        )

        plt.savefig(
            os.path.join(
                figure_dir,
                "05_failure_rate_by_category.pdf"
            ),
            bbox_inches="tight",
        )

        plt.close()


    # ============================================================
    # SUMMARY REPORT
    # ============================================================

    report_lines = []

    report_lines.append(
        "04 - PERFORMANCE DETERMINANTS ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        f"Prediction rows: {len(df)}"
    )

    report_lines.append(
        f"Structures: {df['pdb_id'].nunique()}"
    )

    report_lines.append(
        f"Predictors: {df['predictor'].nunique()}"
    )

    report_lines.append("")

    report_lines.append(
        "ANALYSES PERFORMED"
    )

    report_lines.append(
        "1. Spearman correlations between performance and structural variables"
    )

    report_lines.append(
        "2. Spearman correlations stratified by category"
    )

    report_lines.append(
        "3. Performance by structure size group"
    )

    report_lines.append(
        "4. Performance by chain-count group"
    )

    report_lines.append(
        "5. Multivariable linear regression"
    )

    report_lines.append(
        "6. Failure rates by predictor and category"
    )

    report_lines.append(
        "7. Failure vs structural complexity"
    )

    report_lines.append(
        "8. Logistic regression of prediction success"
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT:"
    )

    report_lines.append(
        "Failed predictions are not treated as zero performance."
    )

    report_lines.append(
        "Performance correlations and regressions use successful predictions."
    )

    report_lines.append(
        "Failure analysis treats success/failure as a separate outcome."
    )

    report_lines.append(
        "AF3 runtime is excluded from runtime analyses because AF3 predictions "
        "were precomputed/imported."
    )

    report_path = os.path.join(
        output_dir,
        "performance_determinants_results.txt"
    )

    save_text_report(
        report_path,
        "\n".join(report_lines)
    )


    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
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