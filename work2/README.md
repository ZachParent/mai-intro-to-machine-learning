# How to Run the Project

This markdown file outlines the requirements, project structure and how to run this project.

## Requirements

- Virtual environment with Python version 3.9

## Project Structure

```
project/
├── data/
|   ├── cross_validated_results/
|   ├── per_fold_results/
|   ├── preprocessed/
│   ├── raw/
├── models/
├── notebooks/
├── tools/
│   ├── analysis/
|   |   ├── __init__.py
|   |   ├── run_statistical_analysis.py
|   |   ├── statistical_analysis_tools.py
│   ├── __init__.py
│   ├── distance.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── reduction.py
│   ├── run.py
│   └── voting.py
│   └── weighting.py
├── requirements.txt
└── README.md
└── HOW_TO_RUN.md (may be combined with README.md)
└── Makefile
└── pyproject.toml
└── setup.cfg
```

## Basic Usage
To run one dataset:
`python tools/run.py --dataset_name [mushroom|hepatitis]`

To perform statistical analysis:
`python tools/analysis/run_statistical_analysis.py`

## Command Line Options
- `--dataset_name`: Choose dataset (required)
    - `mushroom`: Use mushroom dataset
    - `hepatitis`: Use hepatitis dataset
- `--verbose`, `-v`: Enable verbose output
- `--sample`, `-s`: Limit sample size (e.g., -s 1000)

## Example Commands
TODO: verify these commands
*Basic run with mushroom dataset*
python run.py --dataset_name mushroom

*Run hepatitis dataset with verbose output*
python run.py --dataset_name hepatitis -v

*Run mushroom dataset with limited samples*
python run.py --dataset_name mushroom -s 1000

*Combined options: verbose and sample size*
python run.py --dataset_name hepatitis --verbose --sample 200

# All at once
We employ the use of a `Makefile` for the purpose of streamlining the process of running this project.

Simply run this following command in the terminal window:

`COMMAND TO BE DONE`
