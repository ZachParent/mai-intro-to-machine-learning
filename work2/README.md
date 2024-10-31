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

We will probably use 90% or 95% confidence intervals 

We should choose from RNN, FCNN, GCNN

We should choose from RENN, RNGE, ENNTh

We should choose from Drop1, Drop2, Drop3

No need for IB methods

Mandatory reading: Statistical Comparisons of Classifiers over Multiple Data Sets.

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

# 2024-10-29 S5 Practical

* Sheena modified the reduction methods and reran
  * GCNN
  * ENNTH
  * Drop3
* The report is very important, 5/10 points. The code is 5/10 points.
* We need to make it clear in the README.md how to run the code.
  * This may include multiple scripts
* The report should explain how we made decisions during the coding and analysis.

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


### Questions:
* Should we compare statistical analysis across all models or only between models of the same type?
  5 -10 models at a time
* Should we run SVM with reduced datasets?
  * yes
* It's difficult to find statistical significance for the top models of KNN. To get statistically significant results, should we widen the threshold for P-values or should we use a broader sample (linear sample) of our ranked models?
  * better to use a wider alpha threshold