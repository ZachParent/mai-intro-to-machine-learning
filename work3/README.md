# How to Run the Project

This markdown file outlines the requirements, project structure and how to run this project.

## Team

- Kacper Poniatowski
- Zachary Parent
- Sheena Lang
- Carlos Jiménez Farfán

## Requirements

- Python version 3.9
- Make utility installed (optional, for using Makefile commands)
- Git (for cloning the repository)

## Project Structure

```
work3/
├── data/
|   ├── 0_raw/
│   │   ├── hepatitis.arff
│   │   ├── mushroom.arff
│   │   └── vowel.arff
|   ├── 1_preprocessed/
│   │   ├── hepatitis.csv
│   │   ├── mushroom.csv
│   │   └── vowel.csv
│   ├── 2_clustered/
│   │   ├── ...
│   └── 3_metrics.csv
├── notebooks/
├── references/
├── report/
|   ├── 2-data/
|   ├── 3-methods
|   ├── 4-results-and-analysis/
│   ├── figures/
│   ├── tables/
│   ├── 1-introduction.tex
│   ├── 5-conclusion.tex
│   ├── 6-appendix.tex
│   ├── report.pdf
│   └── report.tex
├── src/
│   ├── __init__.py
│   ├── 1_run_preprocessing.py
│   ├── 2_run_model.py
│   ├── 3_run_metrics.py
│   ├── 4_run_analysis.py
│   └── tools/
│   │   ├── analysis/
│   │   │   ├── plots.py
│   │   │   └── tables.py
│   │   ├── clustering/
│   │   │   ├── __init__.py
│   │   │   ├── kmeans.py
│   │   │   ├── fuzzy_cmeans.py
│   │   │   ├── gmeans.py
│   │   │   ├── global_kmeans.py
│   │   │   ├── optics.py
│   │   │   └── spectral_clustering.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── preprocess.py
├── mai-iml-work3-report.pdf
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.cfg
```

## Getting Started

**1. Create a virtual environment**
Create a virtual environment in the `.venv` directory *(it's important to use Python 3.9; if this is not the default on your machine, be sure to create and activate the environment on your own using Python 3.9)*:
```bash
python3.9 -m venv .venv
```

**2. Activate the virtual environment:**
- On Windows
    ```bash
    .venv\Scripts\activate
    ```

- On MacOS/Linux
    ```bash
    source .venv/bin/activate
    ```

**3. Install the necessary requirements:**
```bash
python3.9 -m pip install -r requirements.txt
```

## Execution of Project

Preprocesses the data, run all of the models for each dataset, outputs metrics and performs analysis to generate plots and tables:
```bash
make all
```

### Execute Individual Scripts

**To run preprocessing on all datasets:**
```bash
python src/1_run_preprocessing.py -v
```

**To run a model for one dataset manually:**
```bash
python src/2_run_model.py --dataset <dataset_name> --model <model_name> -v
```

*Command Line Options*

- `--dataset`: Choose dataset (required)
  - `hepatitis`: Use hepatitis dataset
  - `mushroom`: Use mushroom dataset
  - `synthetic`: Use synthetic dataset
  - `vowel`: Use vowel dataset

- `--model`: Choose model (required)
  - `kmeans`: Standard k-means model
  - `fuzzy_cmeans` Fuzzy c-means model
  - `gmeans` G-means model
  - `global_kmeans` Global k-means model
  - `optics` OPTICS model
  - `spectral_clustering` Spectral clustering model

The clustered data will be saved in the `data/2_clustered/` directory.

**To run metrics:**
```bash
python src/3_run_metrics.py -v
```

This command will generate metrics for all clustered data in the `data/2_clustered/` directory and save them in the `data/3_metrics.csv` file.

**To run analysis:**
```bash
python src/4_run_analysis.py -v
```

This command will perform analysis on the metrics data in the `data/3_metrics.csv` file and save the plots and tables in the `report/figures/` and `report/tables/` directories.

### Example Commands
Here are some example commands to help you get started.

Example 1: Run preprocessing on all datasets
```bash
python src/1_run_preprocessing.py -v
```

Example 2: Run the fuzzy c-means model on the hepatitis dataset
```bash
python src/2_run_model.py --dataset hepatitis --model fuzzy_cmeans -v
```

Example 3: Run the kmeans model on the mushroom dataset
```bash
python src/2_run_model.py --dataset mushroom --model kmeans -v
```

Example 4: Run metrics on all available clustered data
```bash
python src/3_run_metrics.py -v
```

Example 5: Run analysis on all available metrics data
```bash
python src/4_run_analysis.py -v
```
