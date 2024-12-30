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
work4/
work4/
├── data/
|   ├── 0_raw/
│   │   ├── mushroom.arff
│   │   └── vowel.arff
|   ├── 1_preprocessed/
│   │   ├── mushroom.csv
│   │   └── vowel.csv
|   ├── 2_reduced/
│   │   ├── ...
|   ├── 3_clustered/
│   │   ├── ...
|   ├── 4_metrics.csv
|   ├── 5_metrics_plots/
│   │   ├── ...
|   ├── 5_metrics_tables/
│   │   ├── ...
|   ├── 6_visualizations/
│   │   ├── ...
├── notebooks/
├── references/
├── report/
|   ├── 4-results-and-analysis/
│   ├── figures/
│   ├── tables/
│   ├── 1-abstract.tex
|   ├── 2-background-and-related-work.tex
|   ├── 3-methods.tex
│   ├── 5-conclusion.tex
│   ├── 6-appendix.tex
│   ├── report.pdf
│   └── report.tex
├── src/
│   ├── notebooks/
│   ├── 1_run_preprocessing.py
│   ├── 2_run_dimensionality_reduction.py
│   ├── 3_run_clustering.py
│   ├── 4_run_metrics.py
│   ├── 5_run_plot_metrics.py
│   ├── 6_run_visualizations.py
│   ├── 2_run_dimensionality_reduction.py
│   ├── 3_run_clustering.py
│   ├── 4_run_metrics.py
│   ├── 5_run_plot_metrics.py
│   ├── 6_run_visualizations.py
│   └── tools/
│   │   ├── analysis/
│   │   ├── clustering/
│   │   │   ├── __init__.py
│   │   │   ├── global_kmeans.py
│   │   │   ├── kmeans.py
│   │   │   └── optics.py
│   │   ├── dimensionality_reduction/
│   │   │   ├── __init__.py
│   │   │   ├── incremental_pca.py
│   │   │   ├── kernel_pca.py
│   │   │   └── pca.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── preprocess.py
├── mai-iml-work4-report.pdf
├── mai-iml-work4-report.pdf
├── Makefile
├── README.md
└── requirements.txt
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

Run the command below to preprocesses the data, run dimensionality reduction on all datasets, run clustering, and acquire metrics, plots and visualizations.
```bash
make all
```

### Execute Individual Scripts

#### 1. Preprocessing
**To run preprocessing on all datasets:**
```bash
python src/1_run_preprocessing.py 
```

The preprocessed data will be saved in the `data/1_preprocessed/` directory.

#### 2. Dimensionality Reduction
**To run dimensionality reduction for one dataset and method:**
```bash
python src/2_run_dimensionality_reduction.py --dataset <dataset_name> --method <method_name> -v
```
*Command Line Options*
- `--dataset`: Choose dataset (required)
- `--method`: Choose method (required)
  - `pca` PCA method
  - `sklearn_pca` PCA method from sklearn
  - `kernel_pca` Kernel PCA method
  - `incremental_pca` Incremental PCA method

The reduced data will be saved in the `data/2_reduced/` directory.

#### 3. Clustering
**To run clustering for one file and model:**
```bash
python src/3_run_clustering.py --input_file_path <input_file_path> --model <model_name> -v
```

*Command Line Options*
- `--input_file_path`: Choose file path (required)
- `--model`: Choose model (required)
  - `global_kmeans` Global k-means model
  - `optics` OPTICS model

The clustered data will be saved in the `data/3_clustered/` directory.

#### 4. Metrics
**To run metrics for all clustered datasets:**
```bash
python src/4_run_metrics.py -v
```

The metrics will be saved in the `data/4_metrics.csv` file.

#### 5. Plot Metrics
**To create plots for the metrics:**
```bash
python src/5_run_plot_metrics.py -v
```

The plots will be saved in the `data/5_metrics_plots/` directory.
The tables will be saved in the `data/5_metrics_tables/` directory.

#### 6. Visualizations
**To create visualizations for the clustered datasets:**
```bash
python src/6_run_visualizations.py -v
```

The visualizations will be saved in the `data/6_visualizations` directory.

### Example Commands
Here are some example commands to help you get started.

Example 1: Run preprocessing on all datasets
```bash
python src/1_run_preprocessing.py 
```

Example 2: Run PCA dimensionality reduction on the mushroom dataset
```bash
python src/2_run_dimensionality_reduction.py --dataset mushroom --method pca -v
```

Example 3: Run OPTICS clustering on a reduced dataset
```bash
python src/3_run_clustering.py --input_file_path data/2_reduced/mushroom/pca/n_components=2.csv --model optics -v
```

Example 4: Generate metrics for all clustered data
```bash
python src/4_run_metrics.py -v
```

Example 5: Create metrics plots
```bash
python src/5_run_plot_metrics.py -v
```

Example 6: Create visualization plots
```bash
python src/6_run_visualizations.py -v
```