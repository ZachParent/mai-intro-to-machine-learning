# TODO

## 2024-10-21

### Reconcile individual work
- [ ] (@Together) pick and choose code for python module

### Finalize python scripts
- [x] (@SheenaLang) create top level script(s) with args for running training and evaluation
- [x] (@SheenaLang) create three 'hard' reduction techniques
- [x] () sanity check results
- [ ] (@ZachParent) add scripts to makefile
- [ ] (@KacperPon) Document scripts / makefile

### Write report
- [x] (@CaJiFan) create structure with subheadings
- [x] (@SheenaLang) `data`
    - [x] (@SheenaLang) dataset
    - [x] (@SheenaLang) preprocessing 
- [ ] () write `methods`
    - [x] (@KacperPon) `knn`
    - [ ] (@CaJiFan) `reduction`
    - [x] (@ZachParent) `svm`
- [ ] () write `results and analysis`
    - [x] (@KacperPon) `knn`
    - [ ] (@CaJiFan) `reduction`
    - [ ] (@ZachParent) `svm`
    - [ ] (@CaJiFan) `intro to statistical analysis`
    - [ ] (@KacperPon) `statistical analysis`
    - [ ] (@ZachParent) plots
- [ ] () write `conclusion`
- [x] (@KacperPon) write `introduction`

## Visualizations
- [x] (@ZachParent) Show different size of mushroom and hepatitis with class 0 and class 1 in them —> Bubble
- [x] (@ZachParent) For hepatitis show imbalance (and maybe missing values) —> Waffle
- [x] (@ZachParent) For each reduction show how accuracy, size and time are affected in a scatter plot --> Three scatter plots
- [ ] (@ZachParent) Boxplots comparing f1 models intra-model type (need to use linear sampling, must justify in report)
- [ ] (@ZachParent) Boxplots comparing f1 models inter-model type (3 of each)

## Report don't-forgets
- [ ] check README.md
- [ ] check that code runs on clean install
- [ ] check that all plots are generated on run
- [ ] explain why f1 was chosen over accuracy
- [ ] explain how reduction is done in report (using best KNN)
- [ ] explain which groups of models pass the friedman test
  - [ ] explain why linear sampling was used
- [ ] explain why ANOVA doesn't work for our use case
- [ ] explain how we chose the values for the SVM parameters
- [ ] explain how we chose the reduction techniques
- [ ] explain how we chose the additional distance metrics for the KNN
- [ ] include tables of key results
- [ ] discussion precision and recall
- [ ] add references
  - [ ] add references for statistical analysis paper
  - [ ] tools used
