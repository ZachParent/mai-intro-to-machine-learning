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
  