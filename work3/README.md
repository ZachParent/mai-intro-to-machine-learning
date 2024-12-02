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
│   │   ├── synthetic.csv
│   │   └── vowel.csv
│   ├── 2_clustered/
│   │   ├── ...
│   ├── 3_metrics.csv
│   └── datasets/
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
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── 1_run_preprocessing.py
│   │   ├── 2_run_model.py
│   │   ├── 3_run_metrics.py
│   │   └── 4_run_analysis.py
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
### Clone the Repository
First, clone the repository to your local machine:
`git clone https://github.com/ZachParent/mai-intro-to-machine-learning.git`

Then, navigate to the project directory:
`cd mai-iml-work3-KacperPoniatowski-ZacharyParent-SheenaLang-CarlosJimenezFarfan`

### Use Make (Recommended)

The easiest way to get started is using our Makefile commands:

**1. Create a virtual environment**
Create a virtual environment in the `.venv` directory *(it's important to use Python 3.9; if this is not the default on your machine, be sure to create and activate the environment on your own using Python 3.9)*:
```bash
make create_environment
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
make install_requirements
```

## Execution of Project

Preprocesses the data, run all of the models for each dataset, outputs metrics and performs analysis to generate plots and tables:
```bash
make all
```

## Manual Setup (Without Make)
If you prefer to set up the project without using `make`, follow these steps:

**1. Create a Virtual Environment**
Create a virtual environment in the `.venv` directory *(it's important to use Python 3.9; if this is not the default on your machine, be sure to create and activate the environment on your own using Python 3.9)*:
```bash
python -m venv .venv
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

**3. Install Requirements**
Install the required packages:
```bash
pip install -r requirements.txt
```

### Execute Scripts

**To run preprocessing on all datasets:**
```bash
python src/scripts/1_run_preprocessing.py
```

**To run a model for one dataset manually:**
```bash
python src/scripts/2_run_model.py --dataset <dataset_name> --model <model_name>
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

**To run metrics:**
`python src/scripts/3_run_metrics.py`

This command will generate metrics for all clustered data.

**To run analysis:**
`python src/scripts/4_run_analysis.py`

This command will perform analysis on the metrics data.

### Example Commands
Here are some example commands to help you get started.

Example 1: Run preprocessing on all datasets
```bash
python src/scripts/1_run_preprocessing.py
```

Example 2: Run the fuzzy c-means model on the hepatitis dataset
```bash
python src/scripts/2_run_model.py --dataset hepatitis --model fuzzy_cmeans
```

Example 3: Run the kmeans model on the mushroom dataset
```bash
python src/scripts/2_run_model.py --dataset mushroom --model kmeans
```

Example 4: Run metrics on all available clustered data
```bash
python src/scripts/3_run_metrics.py
```

Example 5: Run analysis on all available metrics data
```bash
python src/scripts/4_run_analysis.py
```
