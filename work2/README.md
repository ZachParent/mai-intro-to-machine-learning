# How to Run the Project

This markdown file outlines the requirements, project structure and how to run this project.

## Authors

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
project/
├── data/
|   ├── cross_validated_results/
|   ├── per_fold_results/
|   ├── preprocessed/
│   ├── raw/
│   ├── reduced_X/
│   ├── reduced_y/
├── notebooks/
├── tools/
│   ├── analysis/
|   |   ├── __init__.py
|   |   ├── run_statistical_analysis.py
|   |   ├── run_dataset_analysis.py
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
└── Makefile
└── pyproject.toml
└── setup.cfg
```

## Setup and Installation

Note: these commands are to be ran from the root directory.

### Using Make (Recommended)

The easiest way to get started is using our Makefile commands:

1. Set up the complete environment and run the project:

```bash
make run
```

### Manual Usage

To run one dataset manually:
`python tools/run.py --dataset_name [mushroom|hepatitis]`

To perform statistical analysis:
`python tools/analysis/run_statistical_analysis.py --dataset_name[mushroom|hepatitis]`

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
