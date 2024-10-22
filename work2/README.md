# Work 2 Overview

***Due Date:** 2024-11-03*

**Datasets: https://raco.fib.upc.edu/avisos/attachment?assig=IML-MAI&idAvis=102511&idAdjunt=80773**

deliver python scripts in a pycharm project
folders should be used to organize the various algorithms

only certain packages are allowed
- arff_loader
- numpy
- pandas
- scipy
- matplotlib
- sklearn (for some parts)
- seaborn

you will use a predefined stratified 10-fold cross validation to evaluate the performance of your models

train + test always is the full set of data

**Stratified 10-fold cross validation**:
Each fold has the same proportion of classes as the whole dataset

**Predefined:**
The prof has already created the 10 folds for each dataset

Each model gets run 10 times, and the results are averaged to get a single performance metric.

The evaluation should be with 3 metrics:
- accuracy
- efficiency (runtime)
- storage

you need to read the .arff file

use `scipy.io.arffread` to read the .arff file

## Preprocessing

- features may contain **different ranges**
  - normalize or standardize
- features may have **different types**
  - categorical, numerical, mix-type
- features may be **missing values**
  - use the median (for example) to fill in missing values

## Categorical Features

- **Label encoding**:
  - assign a unique integer to each category
  - `sklearn.preprocessing.LabelEncoder`
- **One-hot encoding**:
  - create a new binary feature for each category
  - e.g. color feature with categories: red, blue, green
  - becomes 3 binary features: is_red, is_blue, is_green
  - `sklearn.preprocessing.OneHotEncoder`
---
# Notes from 2024-10-08

### External libraries

Mostly we should implement our own algorithms, but for some parts we can use external libraries:
- sklearn -> SVM (not for KNN, this should be implemented by ourselves)
- can use a library for weighting functions

### Distance metrics

- Minkowski distance with r=1 -> Manhattan distance
- Minkowski distance with r=2 -> Euclidean distance
- One other of our choice

There are many distance metrics listed in slides [work2/s2 -> slide 13](../slides/SuportWork2_2024_wide_ALL_s2.pdf)

### Weighting function
- We should evaluate two
  - Majority
  - Inverse weighted
  - Sheppard's work
- We should try weighting features as well
  - these should be from 0 to 1
  - e.g.
    - Information Gain
    - ReliefF
      - Relief may contain an error that leads to all 0s for weights

### Voting
- We will use K = 1,3,5,7
- Even still ties are possible
- We need to add a policy to break ties

## Work 3 Focuses
1. Distance metrics
2. K parameter
3. Voting scheme
4. Weighting function

### Efficiency
- Could take hours or days to run
- Could avoid using python vectors to speed up the algorithm
- pandas is more efficient

After these next 2 weeks (until 2024-10-22) we should have chosen the ideal configuration for KNN.

## Questions for Prof
- Should we eliminate features that have no relevance?
  - we should let the feature weighting algorithm decide this, and some may have near zero weight
- Can we use scikit-learn pipelines?
  - yes
- Slicing and preprocessing datasets. Should we merge the dataframes before processing or leave them separate?
  - we can choose
  - let's keep them separate and justify our choice, prevents bleeding

# 2024-10-22 S4 Practical
Be careful when choosing the kernal for SVM.

We must use 1 of each family of reduction techniques.
- Condensation
- Edition
- Hybrid

They can be further divided into:
- Incremental
- Decremental
- Batch
- Mixed

Evaluation of search:
- Filter
- Wrapper

The result of the reduction algorithm is a smaller dataset. It is important to show what the percentage of the original dataset is, and to evaluate the difference in performance (time) of the model with the reduced dataset.

It's possible that the performance (accuracy) of the model is improved, because noisy instances are removed.

After we have several candidate models, we must perform statistical analysis. Among the best models we have, we will treat each fold as a different dataset. (10 datasets per input dataset)

Should be 108 models total.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         `tools` and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tools   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes `tools` a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------