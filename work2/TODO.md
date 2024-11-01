# TODO

## 2024-10-21

### Reconcile individual work

- [x] (@Together) pick and choose code for python module

### Finalize python scripts

- [x] (@SheenaLang) create top level script(s) with args for running training and evaluation
- [x] (@SheenaLang) create three 'hard' reduction techniques
- [x] () sanity check results
- [x] (@ZachParent) add scripts to makefile
- [x] (@KacperPon) Document scripts / makefile

### Write report

- [x] (@CaJiFan) create structure with subheadings
- [x] (@SheenaLang) `data`
  - [x] (@SheenaLang) dataset
  - [x] (@SheenaLang) preprocessing
- [x] () write `methods`
  - [x] (@KacperPon) `knn`
  - [x] (@CaJiFan) `reduction`
  - [x] (@ZachParent) `svm`
- [ ] () write `results and analysis`
  - [x] (@KacperPon) `knn`
  - [ ] (@CaJiFan) `reduction`
  - [x] (@ZachParent) `svm`
  - [ ] (@CaJiFan) `intro to statistical analysis`
  - [x] (@KacperPon) `statistical analysis`
  - [x] (@ZachParent) plots
- [x] () write `conclusion`
- [x] (@KacperPon) write `introduction`

## Visualizations

- [x] (@ZachParent) Show different size of mushroom and hepatitis with class 0 and class 1 in them —> Bubble
- [x] (@ZachParent) For hepatitis show imbalance (and maybe missing values) —> Waffle
- [x] (@ZachParent) For each reduction show how accuracy, size and time are affected in a scatter plot --> Three scatter plots
- [ ] (@ZachParent) Boxplots comparing f1 models intra-model type (need to use linear sampling, must justify in report)
- [ ] (@ZachParent) Boxplots comparing f1 models inter-model type (3 of each)

## Report don't-forgets

- [ ] [ ] (@SheenaLang & @KacperPon) check that all plots are generated on run
- [ ] [ ] (@SheenaLang & @KacperPon) check that code runs on clean install

- [x] (@SheenaLang) check README.md
- [x] (@SheenaLang) Reorder methods reduction after SVM
- [x] (@SheenaLang) Add in data section that space becomes sparse if would Onehotencoding

- [ ] (@CaJiFan) explain how reduction is done in report (using best KNN)
- [ ] (@CaJiFan) explain how we chose the reduction techniques
- [ ] (@CaJiFan) add references

  - [ ] (@CaJiFan) add references for statistical analysis paper
  - [ ] (@CaJiFan) tools used

- [ ] (@KacperPon) explain which groups of models pass the friedman test
  - [ ] (@KacperPon) explain why linear sampling was used
- [x] (@KacperPon) explain how we chose the additional distance metrics for the KNN
- [ ] (@KacperPon) Distribute statistical analysis

- [ ] (@ZachParent) explain how we chose the values for the SVM parameters
- [ ] (@ZachParent) Statistical analysis SVM
- [ ] (@ZachParent) Add table in appendix for short names in plots
- [x] (@ZachParent) Add plots for SVM reduction
- [x] (@ZachParent) Fix latex errors

- [ ] (Everyone) include tables of key results
- [ ] (Everyone) Write conclusion
- [ ] (Everyone) Model comparison for mushroom & hepatitis draw conclusions (e.g. if we have a lot of training data --> select model that has a lower training time)

- [x] explain why f1 was chosen over accuracy
- [x] explain why ANOVA doesn't work for our use case
- [x] discussion precision and recall
