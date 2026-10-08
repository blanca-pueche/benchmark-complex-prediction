# Benchmarking AI-based prediction of biomolecular complex structures across diverse molecular interactions

This repository contains the dataset definition, input sequences, benchmark execution scripts, and analysis code used in:

> Benchmarking AI-based prediction of biomolecular complex structures across diverse molecular interactions

The benchmark evaluates five AI-based structure prediction methods across different types of biomolecular interactions using a common dataset and evaluation framework.

## Benchmark methods

The benchmark includes the following prediction methods:

- AlphaFold 3
- Boltz-2
- Chai
- Protenix
- IntelliFold

## Benchmark dataset

The benchmark consists of 150 experimentally determined biomolecular complexes obtained from the Protein Data Bank (PDB).

The dataset contains 30 structures for each interaction category:

| Interaction type | Number of structures |
|---|---:|
| Protein-Protein | 30 |
| Protein-Antibody | 30 |
| Protein-DNA | 30 |
| Protein-RNA | 30 |
| Protein-Peptide | 30 |
| Total | 150 |

The selected structures were deposited in the PDB between 2 June 2023 and 31 December 2024.

The experimentally determined structures are used as reference structures for evaluating the corresponding predicted complexes.

### Dataset characteristics

The benchmark includes structures determined experimentally by X-ray crystallography and cryo-electron microscopy.

The selected complexes cover a range of molecular sizes and interaction types, allowing the prediction methods to be evaluated under different structural and biological conditions.

Before structural comparison, the PDB reference structures were preprocessed by removing water molecules and HETATM records while retaining the relevant biological polymer chains.

## Repository structure

    benchmark-complex-prediction/
    |
    +-- README.md
    +-- LICENSE
    +-- CITATION.cff
    +-- .gitignore
    +-- requirements.txt
    |
    +-- data/
    |   +-- benchmark_metadata.csv
    |   |
    |   +-- ids/
    |   |   +-- protein_antibody.txt
    |   |   +-- protein_dna.txt
    |   |   +-- protein_peptide.txt
    |   |   +-- protein_protein.txt
    |   |   +-- protein_rna.txt
    |   |
    |   +-- fastas/
    |       +-- protein_antibody/
    |       +-- protein_dna/
    |       +-- protein_peptide/
    |       +-- protein_protein/
    |       +-- protein_rna/
    |
    +-- scripts/
    |   +-- select_ids.py
    |   +-- download_fastas.py
    |   +-- filter_fasta.py
    |   +-- filter_rna_sequences.py
    |   +-- get_metadata.py
    |   +-- create_reference_csv.py
    |   +-- run_benchmark.py
    |   +-- create_results_csv.py
    |
    +-- analysis/
        +-- benchmark_dataset.csv
        +-- 03_statistical_analysis.py
        +-- extreme_case_analysis.py
        +-- ...

## Data

### Benchmark metadata

`data/benchmark_metadata.csv` contains metadata associated with the benchmark structures, including information retrieved from the RCSB PDB and information used for benchmark characterization.

### PDB identifiers

The `data/ids/` directory contains the final PDB identifiers included in each benchmark category.

Each file contains one PDB identifier per line:

- `protein_antibody.txt`
- `protein_dna.txt`
- `protein_peptide.txt`
- `protein_protein.txt`
- `protein_rna.txt`

### FASTA sequences

The `data/fastas/` directory contains the FASTA sequences corresponding to the selected PDB structures.

Sequences are organized according to interaction type:

- `protein_antibody/`
- `protein_dna/`
- `protein_peptide/`
- `protein_protein/`
- `protein_rna/`

## Requirements

The repository contains scripts for dataset preparation, benchmark execution, result extraction, and statistical analysis.

### General Python dependencies

The data preparation and analysis scripts require Python and commonly used scientific Python packages, including:

- NumPy
- pandas
- SciPy
- Matplotlib
- seaborn
- scikit-learn
- statsmodels
- requests

The corresponding dependencies can be installed using:

    pip install -r requirements.txt

### Structural prediction software

Running the complete benchmark additionally requires the prediction methods and structural-analysis software used in the study:

- AlphaFold 3
- Boltz-2
- Chai
- Protenix
- IntelliFold
- Scipion
- OpenStructure

The prediction methods have their own software, Python, CUDA, GPU, and/or system requirements. These environments are not included in this repository and should be installed according to the documentation of the respective software.

## Dataset preparation

The scripts in `scripts/` can be used to prepare and inspect the benchmark dataset.

### Selecting PDB identifiers

PDB identifiers can be randomly selected from an input list:

    python scripts/select_ids.py \
        --input <input_ids.txt> \
        --number 30 \
        --output <selected_ids.txt>

An optional random seed can be specified for reproducibility:

    python scripts/select_ids.py \
        --input <input_ids.txt> \
        --number 30 \
        --output <selected_ids.txt> \
        --seed 42

Previously selected identifiers can also be excluded:

    python scripts/select_ids.py \
        --input <input_ids.txt> \
        --number 30 \
        --output <selected_ids.txt> \
        --exclude <already_selected.txt> \
        --seed 42

### Downloading FASTA sequences

FASTA sequences can be downloaded directly from the RCSB PDB:

    python scripts/download_fastas.py \
        --input <selected_ids.txt> \
        --output <output_directory>

Existing FASTA files can be overwritten using:

    python scripts/download_fastas.py \
        --input <selected_ids.txt> \
        --output <output_directory> \
        --overwrite

### Filtering FASTA files

The FASTA filtering script can be used to remove unwanted structures and optionally retain a defined number of entries:

    python scripts/filter_fasta.py \
        --input <fasta_directory> \
        --exclude-folder <comparison_directory> \
        --number 30 \
        --seed 42

### Filtering RNA-containing structures

RNA sequences can be filtered using:

    python scripts/filter_rna_sequences.py \
        --input <protein_rna.txt> \
        --output <protein_rna_filtered.txt>

The script checks the sequences retrieved from the RCSB PDB and applies the RNA sequence criteria used during benchmark dataset preparation.

### Retrieving PDB metadata

Metadata can be retrieved from the RCSB PDB using:

    python scripts/get_metadata.py \
        --input-dir <fastas_directory> \
        --output <metadata.csv>

## Running the benchmark

The benchmark prediction workflow is implemented using Scipion.

The benchmark runner accepts the required input directories and configuration through command-line arguments rather than relying on machine-specific paths.

For example:

    scipion3 python scripts/run_benchmark.py \
        --fasta-dir <fastas_directory> \
        --af3-dir <af3_predictions_directory> \
        --log-file <benchmark_status.tsv>

The benchmark runner creates the corresponding Scipion workflows for the selected structures and prediction methods.

The workflow includes:

1. Importing the experimentally determined reference structure.
2. Preparing the reference structure.
3. Importing AlphaFold 3 predictions.
4. Running Boltz-2 predictions.
5. Running Chai predictions.
6. Running Protenix predictions.
7. Running IntelliFold predictions.
8. Comparing predicted structures with the experimental reference using OpenStructure.
9. Generating structural discrepancy information for further analysis.

The exact computational requirements depend on the prediction methods being executed and the available CPU/GPU resources.

## Extracting benchmark results

After the Scipion projects have been completed, the structural comparison results can be collected into a single CSV file.

    scipion3 python scripts/create_results_csv.py \
        --projects-dir <scipion_projects_directory> \
        --output <results.csv>

The resulting table contains the structural evaluation metrics and prediction status for the different methods and benchmark structures.

Failed predictions are retained as failures rather than being assigned artificial metric values.

## Structural evaluation

Predicted structures are evaluated against their corresponding experimentally determined reference structures using OpenStructure.

The primary structural evaluation metrics are:

- lDDT
- TM-score
- RMSD
- DockQ

Additional interface and structural-quality metrics are also calculated, including:

- Backbone lDDT
- Global QS-score
- Best QS-score
- DockQ variants
- Oligomeric GDT-TS
- Oligomeric GDT-HA
- Number of clashes
- Number of bad bonds
- Number of bad angles

The same reference structure and evaluation procedure are used across prediction methods.

## Statistical analysis

The statistical analysis can be run using:

    python analysis/03_statistical_analysis.py \
        --input <benchmark_dataset.csv> \
        --output-dir <output_directory>

The analysis compares prediction methods across the benchmark categories and evaluation metrics.

For the primary structural metrics, complete cases across the prediction methods are used for paired statistical comparisons.

The analysis includes:

- Friedman tests for overall differences between prediction methods.
- Pairwise Wilcoxon signed-rank tests.
- Benjamini-Hochberg correction for multiple comparisons.
- Rank-biserial effect sizes.
- McNemar tests for prediction success rates.
- Runtime comparisons for locally executed prediction methods.
- Analysis of associations between prediction performance and structural characteristics.

AlphaFold 3 prediction runtime is excluded from runtime comparisons because AlphaFold 3 predictions were generated using the web server rather than through the local benchmark workflow.

## Extreme-case analysis

The extreme-case analysis identifies benchmark structures showing large disagreement between prediction methods.

It can be run using:

    python analysis/06_extreme_case_analysis.py \
        --input <benchmark_dataset.csv> \
        --scipion-projects <scipion_projects_directory> \
        --output-dir <output_directory>

Structure-level disagreement is calculated from the variation between prediction methods across the primary structural metrics.

The analysis then performs residue-level investigation of selected extreme cases using the structural discrepancy information generated by the Scipion workflows.

The analysis includes:

- Structure-level disagreement scoring.
- Selection of extreme benchmark cases.
- Residue-level prediction deviations.
- Rolling-window analysis.
- Identification of regions with high prediction disagreement.
- Predictor-specific comparisons.
- Residue-level visualization.
- Predictor comparison heatmaps.

## Reproducibility

The repository is intended to provide the information required to reproduce the benchmark dataset preparation and downstream analysis.

The repository includes:

- Final benchmark PDB identifiers.
- FASTA input sequences.
- Benchmark metadata.
- Dataset preparation scripts.
- Benchmark execution scripts.
- Structural result extraction scripts.
- Statistical analysis scripts.
- Extreme-case analysis scripts.

Large prediction outputs, Scipion project directories, intermediate files, and other computationally generated data are not included in the repository.

The benchmark workflows can therefore be reproduced using the provided inputs and scripts together with the required external prediction software.

## AlphaFold 3

AlphaFold 3 predictions were generated using the AlphaFold 3 web server.

The AlphaFold 3 predictions used in the benchmark were therefore obtained independently of the local Scipion execution environment.

Consequently, AlphaFold 3 prediction runtime is not included in comparisons of local prediction runtimes.

## PDB data

Experimental structures and associated metadata are derived from the Protein Data Bank (PDB).

PDB identifiers are provided in this repository so that the original experimental structures can be retrieved from the RCSB PDB.

The PDB and RCSB PDB have their own terms of use and data policies, which apply to the corresponding structural data.

## Software and external resources

This benchmark relies on several external software packages and resources:

- AlphaFold 3
- Boltz-2
- Chai
- Protenix
- IntelliFold
- Scipion
- OpenStructure
- RCSB Protein Data Bank

Please cite the original publications and software resources associated with these tools when using this repository.

## License

The source code in this repository is distributed under the license specified in `LICENSE`.

The external datasets and software used by the benchmark remain subject to their respective licenses and terms of use.

## Citation

If you use this benchmark dataset, code, or analysis workflow, please cite:

> Pueche-Granados, B. and Sorzano, C. O. S.
> *Benchmarking AI-based prediction of biomolecular complex structures across diverse molecular interactions.*

The repository citation metadata is provided in `CITATION.cff`.

Once the repository has been archived through Zenodo, the DOI will also be provided here.

## Contact

For questions, suggestions, or issues related to the benchmark or repository, please open an issue in this repository.

