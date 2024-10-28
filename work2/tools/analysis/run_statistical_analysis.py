# %%
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math
import argparse
from tools.analysis.statistical_analysis_tools import *

plt.style.use('default')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/per_fold_results")
parser.add_argument("--figures_dir", type=str, default="reports/figures")
args = parser.parse_args()

DATA_DIR = args.data_dir
FIGURES_DIR = args.figures_dir

# %%
knn_results_hepatitis = pd.read_csv(f"{DATA_DIR}/knn_hepatitis.csv")
# knn_results_mushroom = pd.read_csv(f"{DATA_DIR}/knn_mushroom.csv")
knn_reduction_results_hepatitis = pd.read_csv(f"{DATA_DIR}/knn_reduction_hepatitis.csv")
# knn_reduction_results_mushroom = pd.read_csv(f"{DATA_DIR}/knn_reduction_mushroom.csv")
svm_results_hepatitis = pd.read_csv(f"{DATA_DIR}/svm_hepatitis.csv")
# svm_results_mushroom = pd.read_csv(f"{DATA_DIR}/svm_mushroom.csv")

# %%
fold_cols = [f'fold{i}' for i in range(1, 11)]

for df in [knn_results_hepatitis, knn_reduction_results_hepatitis, svm_results_hepatitis]:
    df['mean_f1_score'] = df.loc[:, fold_cols].mean(axis=1)
    df['std_f1_score'] = df.loc[:, fold_cols].std(axis=1)



# %%
num_sample_options = [4, 8, 16]
sample_types = ['linear', 'top']
models = ['KNN', 'SVM', 'KNN-Reduction']
dataframes = [knn_results_hepatitis, svm_results_hepatitis, knn_reduction_results_hepatitis]
data = []

for sample_type, num_samples, (model, df) in itertools.product(sample_types, num_sample_options, zip(models, dataframes)):
    sample_func = linear_sample if sample_type == 'linear' else top_samples
    
    p_value = friedman_test(sample_func(df, num_samples), fold_cols)[1]
    data.append({
        'model': model,
        'num_samples': num_samples,
        'sample_type': sample_type,
        'p_value': p_value,
        'significant': p_value < 0.05
    })

p_values_df = pd.DataFrame(data)

p_values_df

# %%
# TODO: only plot one point for KNN-Reduction

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
fig.suptitle('P-values vs Number of Samples for Different Models and Sampling Methods')

plot_p_values_vs_num_samples(axes, p_values_df, models, sample_types, num_sample_options)

plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/p_values_vs_num_samples.png', dpi=300)
plt.show()


# %%
fig = plot_independent_effects(knn_results_hepatitis, ['k', 'distance_func', 'voting_func', 'weighting_func'])
fig.suptitle('Independent Effects of KNN Model Parameters on F1 Score', fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.9)
fig.savefig(f'{FIGURES_DIR}/independent_effects_KNN.png', dpi=300)
plt.show()


# %%
fig = plot_independent_effects(svm_results_hepatitis, ['C', 'kernel_type'])
fig.suptitle('Independent Effects of SVM Model Parameters on F1 Score', fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.85)
fig.savefig(f'{FIGURES_DIR}/independent_effects_SVM.png', dpi=300)
plt.show()

# %%
fig = plot_interactions(knn_results_hepatitis, ['k', 'distance_func', 'voting_func', 'weighting_func'])
fig.suptitle('Interaction Effects of KNN Model Parameters on F1 Score', fontsize=20, fontweight='bold')
fig.subplots_adjust(top=0.95)
fig.savefig(f'{FIGURES_DIR}/interaction_effects_KNN.png', dpi=300)
plt.show()

knn_col_names = ['k', 'distance_func', 'voting_func', 'weighting_func']
top_values = get_top_values(knn_results_hepatitis, knn_col_names, num_to_select=2, models_to_consider=10)

models_with_top_values = get_models_with_top_values(knn_results_hepatitis, top_values)


# %%
nemenyi_results = nemenyi_test(models_with_top_values, fold_cols)
nemenyi_results


# %%

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(nemenyi_results, fmt='.2f', annot=True, cmap='coolwarm', ax=ax, cbar=False)
fig.suptitle('Nemenyi Test Results for KNN Models with Top Values', fontsize=20, fontweight='bold')
fig.savefig(f'{FIGURES_DIR}/nemenyi_test_results_KNN.png', dpi=300)
plt.tight_layout()
plt.show()

analyze_parameters(models_with_top_values, nemenyi_results)

significant_pairs_df = get_significant_pairs(nemenyi_results, models_with_top_values)



