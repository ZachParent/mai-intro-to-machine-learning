# How to Run the Project

This markdown file outlines the requirements, project structure and how to run this project.

## Authors
- Kacper Poniatowski
- Zachary Parent
- Sheena Lang
- Carlos Jiménez Farfán 🐍

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
├── models/
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

### Statistical Analysis Explanation

* ANOVA is suitable at times, but not for our case.
  * Because it's parametric
  * Explain with an excerpt from the slides
* Friedman is suitable for our case.
  * Compares the mean ranks of the models (not the accuracies themselves)
  * Explain with an excerpt from the slides
* Then we compute the post-hoc test/
  * Nemenyi is used for pairwise comparisons.
  * Explain with an excerpt from the slides
  * When ranking, it's important to ensure that ties are treated as 2.5, 2.5, insteado of 2, 3, e.g.
* The whole bar is the CD. CD/2 is the half-width.
  * If the CD/2 doesn't overlap, the difference is statistically significant.
  * We should plot either the way in the reading, or with whisker plots.
  * Section 4 onward is the important part (Empirical comparison of tests)
    * Important to understand the tools, not the formulas

### TODO:
* Check if the weighting functions are working correctly
* Create analysis for time and storage complexity, especially comparing across model types
* We should cross-read each other's report parts before submitting
* Finish statistical analysis.
* Write up about statistical analysis

* Zach to check information gain weighting functions
* Carlos to check reduction techniques
  * Make the reduction data use sums for each metric
* Carlos to write theory on statistical analysis  
* Carlos to write about reduction for SVM
* Kacper to write up about statistical analysis results
* Sheena to implement reduction techniques for SVM
* Sheena to write about time and storage complexity analysis
* Zach to run statistical analysis scripts and organize repo
This command will: 
- Create a virtual environment
- Install all requirements
- Format the code
- Prepare the project for running
- Ensure the environment is created
- Run both datasets (mushroom and hepatitis) through the main analysis
- Run statistical analysis for both datasets
- Run dataset analysis
- Save results in the appropriate directories


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
*Basic run with mushroom dataset*
python tools/run.py --dataset_name mushroom

*Run hepatitis dataset with verbose output*
python tools/run.py --dataset_name hepatitis -v

*Run mushroom dataset with limited samples*
python tools/run.py --dataset_name mushroom -s 1000

*Combined options: verbose and sample size*
python tools/run.py --dataset_name hepatitis --verbose --sample 200
