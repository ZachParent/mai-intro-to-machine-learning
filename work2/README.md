# How to Run the Project

This markdown file outlines the requirements, project structure and how to run this project.

## Team

- Kacper Poniatowski
- Zachary Parent
- Sheena Lang
- Carlos Jiménez Farfán

## Requirements

- Python version 3.9
- Make utility installed
- Git (for cloning the repository)

## Project Structure

```
work2/
├── data/
|   ├── cross_validated_results/
|   ├── per_fold_results/
│   ├── raw/
│   ├── reduced_X/
│   ├── reduced_y/
├── notebooks/
├── references/
├── reports/
|   ├── 0.2-data/
|   ├── 0.3-methods
|   ├── 0.4-results-and-analysis/
│   ├── figures/
│   ├── tables/
│   ├── 01-introduction.tex
│   ├── 05-conclusion.tex
│   ├── 06-appendix.tex
│   ├── report.pdf
│   ├── report.tex
├── tools/
│   ├── analysis/
|   |   ├── __init__.py
|   |   ├── run_dataset_analysis.py
|   |   ├── run_statistical_analysis.py
|   |   ├── statistical_analysis_tools.py
|   |   ├── table_tool.py
│   ├── __init__.py
│   ├── distance.py
│   ├── knn.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── reduction.py
│   ├── run.py
│   └── voting.py
│   └── weighting.py
└── Makefile
└── pyproject.toml
└── README.md
├── requirements.txt
└── setup.cfg
```

## Running script for the first time

Note: these commands are to be ran from the root directory.

### Use Make (Recommended)

The easiest way to get started is using our Makefile commands:

1. Create a virtual environment:

```bash
make create_environment
```

2. Activate the virtual environment:
```bash
.venv\Scripts\activate
```
or
```bash
source .venv/bin/activate
```

3. Install the necessary requirements:
```bash
make install_requirements
```

## Execute scripts

4. Run all models and corresponding analysis:
```bash
make all
```

## Details

### Manual Usage

To run one dataset manually:
`python tools/run.py --dataset_name [mushroom|hepatitis`

To perform statistical analysis:
`python tools/analysis/run_statistical_analysis.py --dataset_name [mushroom|hepatitis]`

**Command Line Options**

- `--dataset_name`: Choose dataset (required)
  - `mushroom`: Use mushroom dataset
  - `hepatitis`: Use hepatitis dataset
- `--verbose`, `-v`: Enable verbose output
- `--sample`, `-s`: Limit sample size (e.g., -s 1000)

### Example Commands

_Basic run with mushroom dataset_
python tools/run.py --dataset_name mushroom

_Run hepatitis dataset with verbose output_
python tools/run.py --dataset_name hepatitis -v

_Run mushroom dataset with limited samples_
python tools/run.py --dataset_name mushroom -s 1000

_Combined options: verbose and sample size_
python tools/run.py --dataset_name hepatitis --verbose --sample 200
