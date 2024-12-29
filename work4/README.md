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
├── data/
|   ├── 0_raw/
│   │   ├── mushroom.arff
│   │   └── vowel.arff
|   ├── 1_preprocessed/
│   │   ├── mushroom.csv
│   │   └── vowel.csv
│   ├── 3_clustered/
│   │   ├── ...
│   └── 3_metrics.csv
│   └── 4_metrics.csv
├── references/
├── report/
|   ├── 4-results-and-analysis/
│   ├── figures/
│   ├── tables/
│   ├── 1-abstract.tex
│   ├── 2-background-and-related-work.tex
│   ├── 3-methods.tex
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
│   └── tools/
│   │   ├── analysis/
│   │   ├── clustering/
│   │   ├── dimensionality_reduction/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── preprocess.py
├── mai-iml-work4-report.pdf
├── Makefile
├── README.md
├── requirements.txt
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

**To run preprocessing on all datasets:**
```bash
python src/1_run_preprocessing.py 
```

**To run dimensionality reduction:**
```bash
python src/2_run_dimensionality_reduction.py --dataset <dataset_name> --method <method_name>
```

*Command Line Options*

- `--dataset`: Choose dataset (required)
  - `mushroom`: Use mushroom dataset
  - `vowel`: Use vowel dataset

- `--method`: Choose reduction method (required)
  - `pca`: Use PCA
  - `sklearn_pca`: Use sklearn PCA
  - `kernel_pca`: Use kernel PCA
  - `incremental_pca`?: Use incremental PCA

**To run clustering:**
```bash
python src/3_run_clustering.py
```

*Command Line Options*

- `--model`: Choose model (required)
  - `global_kmeans`: Use global k-means
  - `optics`: Use optics

- `--input_file_path`: Input file path (required)

- `--reduced`: Whether the input data is reduced

**To run metrics:**
```bash
python src/4_run_metrics.py
```

**To run plot metrics:**
```bash
python src/5_run_plot_metrics.py
```

**To run visualisations:**
```bash
python src/6_run_visualizations.py
```

### Example Commands
Here are some example commands to help you get started.

Example 1: Run preprocessing on all datasets
```bash
python src/1_run_preprocessing.py 
```

Example 2: Run metrics on all available data
```bash
python src/4_run_metrics.py 
```

Example 3: Run plot metrics on all available data
```bash
python src/5_run_plot_metrics.py 
```

Example 4: Run visualisations on all available data
```bash
python src/6_run_visualizations.py
```


