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
from analysis.statistical_analysis_tools import *
from analysis.table_tool import *
import os
import logging

# %%
from pypalettes import load_cmap

alexandrite = load_cmap("Alexandrite")
emrld = load_cmap("Emrld", reverse=True)

# %%

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "../../reports/figures")
TABLES_DIR = os.path.join(SCRIPT_DIR, "../../reports/tables")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="hepatitis")
parser.add_argument("--f", type=str, default="")
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
dataset_name = args.dataset_name

# %%
logging.info(f"Loading results data for {dataset_name}")
knn_results_filename = f"{DATA_DIR}/per_fold_results/knn_{dataset_name}.csv"
knn_reduction_results_filename = f"{DATA_DIR}/per_fold_results/knn_reduction_{dataset_name}.csv"
svm_results_filename = f"{DATA_DIR}/per_fold_results/svm_{dataset_name}.csv"
svm_reduction_results_filename = f"{DATA_DIR}/per_fold_results/svm_reduction_{dataset_name}.csv"

knn_results = pd.read_csv(knn_results_filename)
knn_reduction_results = pd.read_csv(knn_reduction_results_filename)
svm_results = pd.read_csv(svm_results_filename)
svm_reduction_results = pd.read_csv(svm_reduction_results_filename)

f1_cols = [f"f1_{i}" for i in range(10)]
train_time_cols = [f"train_time_{i}" for i in range(10)]
test_time_cols = [f"test_time_{i}" for i in range(10)]
storage_cols = [f"storage_{i}" for i in range(10)]
knn_col_names = ["k", "distance_func", "voting_func", "weighting_func"]
svm_col_names = ["C", "kernel_type"]

# %%
knn_results["model_label"] = knn_results.apply(get_knn_model_label, axis=1)
knn_reduction_results["model_label"] = knn_reduction_results.apply(
    get_knn_reduction_model_label, axis=1
)
svm_results["model_label"] = svm_results.apply(get_svm_model_label, axis=1)
svm_reduction_results["model_label"] = svm_reduction_results.apply(
    get_svm_reduction_model_label, axis=1
)

knn_legend = knn_results.loc[:, ["model_label"] + knn_col_names]
svm_legend = svm_results.loc[:, ["model_label"] + svm_col_names]
knn_reduction_legend = knn_reduction_results.loc[:, ["model_label"] + knn_col_names]
svm_reduction_legend = svm_reduction_results.loc[:, ["model_label"] + svm_col_names]

for df, name in zip(
    [knn_legend, svm_legend, knn_reduction_legend, svm_reduction_legend],
    ["KNN", "SVM", "KNN-Reduction", "SVM-Reduction"],
):
    df = format_column_names(df)
    write_latex_table(
        df,
        f"{TABLES_DIR}/{name}_legend.tex",
        f"{name} Legend",
        longtable=df.shape[0] > 20,
    )
# %%
for df in [knn_results, knn_reduction_results, svm_results, svm_reduction_results]:
    for metric_col in ["f1", "train_time", "test_time"]:
        df[f"mean_{metric_col}"] = df.loc[:, [f"{metric_col}_{i}" for i in range(10)]].mean(axis=1)
    df.sort_values(by=f"mean_f1", ascending=False, inplace=True)
knn_reduction_results["mean_storage"] = knn_reduction_results[storage_cols].mean(axis=1)
svm_reduction_results["mean_storage"] = svm_reduction_results[storage_cols].mean(axis=1)

# %%
for df, name, filename in zip(
    [knn_results, svm_results, knn_reduction_results, svm_reduction_results],
    ["KNN", "SVM", "KNN-Reduction", "SVM-Reduction"],
    ["knn_results", "svm_results", "knn_reduction_results", "svm_reduction_results"],
):
    print(f"{df} has mean_f1: {df['mean_f1']}")
    write_latex_table_summary(
        df,
        ["model_label", "mean_f1", "mean_train_time", "mean_test_time"]
        + (["mean_storage"] if "reduction" in filename else []),
        f"{TABLES_DIR}/{filename}_{dataset_name}.tex",
        f"Results from {name} models for the {dataset_name} dataset".title(),
        sort_by="mean_f1",
    )

# %%
logging.info("Running Friedman test across all models and sampling methods")
num_sample_options = [4, 8, 16]
sample_types = ["linear", "top"]
models = ["KNN", "SVM", "KNN-Reduction", "SVM-Reduction"]
dataframes = [knn_results, svm_results, knn_reduction_results, svm_reduction_results]
data = []

for sample_type, num_samples, (model, df) in itertools.product(
    sample_types, num_sample_options, zip(models, dataframes)
):
    sample_func = linear_sample if sample_type == "linear" else top_samples

    p_value = friedman_test(sample_func(df, num_samples), f1_cols)[1]
    data.append(
        {
            "model": model,
            "num_samples": num_samples,
            "sample_type": sample_type,
            "p_value": 1 if math.isnan(p_value) else p_value,
            "significant": p_value < 0.05,
        }
    )

p_values_df = pd.DataFrame(data)

p_values_df

# %%
logging.info("Plotting p-values vs number of samples")
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)

plot_p_values_vs_num_samples(axes, p_values_df, models, sample_types, num_sample_options)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/p_values_vs_num_samples_{dataset_name}.png", dpi=300)
plt.show()


# %%
logging.info("Plotting independent effects of KNN model parameters")
fig = plot_independent_effects(
    knn_results, ["k", "distance_func", "voting_func", "weighting_func"]
)

fig.savefig(f"{FIGURES_DIR}/independent_effects_KNN_{dataset_name}.png", dpi=300)
plt.show()


# %%
logging.info("Plotting independent effects of SVM model parameters")
fig = plot_independent_effects(svm_results, ["C", "kernel_type"])
fig.savefig(f"{FIGURES_DIR}/independent_effects_SVM_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Plotting interaction effects of KNN model parameters")
fig = plot_interactions(knn_results, ["k", "distance_func", "voting_func", "weighting_func"])
fig.savefig(f"{FIGURES_DIR}/interaction_effects_KNN_{dataset_name}.png", dpi=300)
plt.show()

top_values = get_top_values(knn_results, knn_col_names, num_to_select=2, models_to_consider=10)

models_with_top_values = get_models_with_top_values(knn_results, top_values)

# %%
logging.info("Plotting interaction effects of SVM model parameters")
fig = plot_interactions(svm_results, ["C", "kernel_type"])
fig.savefig(f"{FIGURES_DIR}/interaction_effects_SVM_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Plotting ranked folds distribution for KNN models")
knn_ranked_folds = get_ranked_folds(linear_sample(knn_results, 8), f1_cols)
fig, ax = plt.subplots(figsize=(12, 6))
plot_ranked_folds(ax, knn_ranked_folds, f1_cols)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/ranked_folds_KNN_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Plotting ranked folds distribution for SVM models")
svm_ranked_folds = get_ranked_folds(linear_sample(svm_results, 8), f1_cols)
fig, ax = plt.subplots(figsize=(12, 6))
plot_ranked_folds(ax, svm_ranked_folds, f1_cols)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/ranked_folds_SVM_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Running Nemenyi test for KNN models")
knn_results_for_nemenyi = linear_sample(knn_results, 8)
knn_nemenyi_results = nemenyi_test(knn_results_for_nemenyi, f1_cols)
knn_nemenyi_results


# %%
logging.info("Plotting Nemenyi test results for KNN models")
fig, ax = plt.subplots(figsize=(12, 12))
model_labels = knn_results_for_nemenyi["model_label"]
sns.heatmap(
    knn_nemenyi_results,
    fmt=".2f",
    annot=True,
    cmap=emrld,
    ax=ax,
    cbar=False,
    xticklabels=model_labels,
    yticklabels=model_labels,
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/nemenyi_test_results_KNN_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Analyzing Nemenyi test results for KNN models")
analyze_parameters(knn_results_for_nemenyi, knn_nemenyi_results, knn_col_names)
significant_pairs = get_significant_pairs(knn_nemenyi_results)
significant_pairs_df = get_df_pairs(knn_results_for_nemenyi, significant_pairs).loc[
    :, knn_col_names + ["mean_f1"]
]
significant_pairs_df = format_column_names(significant_pairs_df)
write_latex_table(
    significant_pairs_df,
    f"{TABLES_DIR}/knn_significant_pairs_{dataset_name}.tex",
    f"Significant Differences in KNN Models for {dataset_name.title()}",
)
# %%
logging.info("Running Nemenyi test for SVM models")
svm_results_for_nemenyi = linear_sample(svm_results, 8)
svm_nemenyi_results = nemenyi_test(svm_results_for_nemenyi, f1_cols)
svm_nemenyi_results


# %%
logging.info("Plotting Nemenyi test results for SVM models")
fig, ax = plt.subplots(figsize=(12, 12))
model_labels = svm_results_for_nemenyi["model_label"]
sns.heatmap(
    svm_nemenyi_results,
    fmt=".2f",
    annot=True,
    cmap=emrld,
    ax=ax,
    cbar=False,
    xticklabels=model_labels,
    yticklabels=model_labels,
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/nemenyi_test_results_SVM_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Analyzing Nemenyi test results for SVM models")
analyze_parameters(svm_results_for_nemenyi, svm_nemenyi_results, svm_col_names)

significant_pairs = get_significant_pairs(svm_nemenyi_results)
significant_pairs_df = get_df_pairs(svm_results_for_nemenyi, significant_pairs)[
    svm_col_names + ["mean_f1"]
]
significant_pairs_df = format_column_names(significant_pairs_df)
write_latex_table(
    significant_pairs_df,
    f"{TABLES_DIR}/svm_significant_pairs_{dataset_name}.tex",
    f"Significant Differences in SVM Models for {dataset_name.title()}",
)
# %%
logging.info("Running Nemenyi test for SVM-Reduction models")
svm_reduction_results_for_nemenyi = svm_reduction_results
svm_reduction_nemenyi_results = nemenyi_test(svm_reduction_results_for_nemenyi, f1_cols)
svm_reduction_nemenyi_results


# %%
logging.info("Plotting Nemenyi test results for SVM-Reduction models")
fig, ax = plt.subplots(figsize=(12, 12))
model_labels = svm_reduction_results_for_nemenyi["model_label"]
sns.heatmap(
    svm_reduction_nemenyi_results,
    fmt=".2f",
    annot=True,
    cmap=emrld,
    ax=ax,
    cbar=False,
    xticklabels=model_labels,
    yticklabels=model_labels,
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/nemenyi_test_results_SVM_Reduction_{dataset_name}.png", dpi=300)
plt.show()

# %%
logging.info("Analyzing Nemenyi test results for SVM-Reduction models")
analyze_parameters(svm_reduction_results_for_nemenyi, svm_reduction_nemenyi_results, svm_col_names)

significant_pairs = get_significant_pairs(svm_reduction_nemenyi_results)
significant_pairs_df = get_df_pairs(svm_reduction_results_for_nemenyi, significant_pairs)[
    ["reduction_func"] + svm_col_names + ["mean_f1"]
]
significant_pairs_df = format_column_names(significant_pairs_df)
write_latex_table(
    significant_pairs_df,
    f"{TABLES_DIR}/svm_reduction_significant_pairs_{dataset_name}.tex",
    f"Significant Differences in SVM-Reduction Models for {dataset_name.title()}",
)
# %%

best_svm_model = svm_results.iloc[0, :]
best_knn_model = knn_results.iloc[0, :]
knn_svm_f1_p_value = (
    stats.wilcoxon(best_svm_model[f1_cols].to_list(), best_knn_model[f1_cols].to_list()).pvalue
    if not all(best_svm_model[f1_cols] == best_knn_model[f1_cols])
    else 1
)
knn_svm_train_time_p_value = stats.wilcoxon(
    best_svm_model[train_time_cols].to_list(), best_knn_model[train_time_cols].to_list()
).pvalue
knn_svm_test_time_p_value = stats.wilcoxon(
    best_svm_model[test_time_cols].to_list(), best_knn_model[test_time_cols].to_list()
).pvalue
svm_knn_comparison_df = pd.DataFrame(
    {
        "metric": ["F1 Score", "Train Time", "Test Time"],
        "p_value": [knn_svm_f1_p_value, knn_svm_train_time_p_value, knn_svm_test_time_p_value],
    }
)
write_latex_table(
    format_column_names(svm_knn_comparison_df),
    f"{TABLES_DIR}/svm_knn_wilcoxon_comparison_{dataset_name}.tex",
    f"Wilcoxon Signed-Rank Test Results for SVM and KNN Models for {dataset_name.title()}",
)
# %%
metric_cols_array = np.array([f1_cols, train_time_cols, test_time_cols])
all_metric_cols = metric_cols_array.flatten().tolist()
best_knn_and_svm = pd.DataFrame(
    [
        ["KNN"] + best_knn_model[all_metric_cols].to_list(),
        ["SVM"] + best_svm_model[all_metric_cols].to_list(),
    ],
    columns=["model"] + all_metric_cols,
)
best_knn_and_svm
# %%
# Expand and plot the data
best_knn_and_svm_by_fold = expand_data_per_fold(best_knn_and_svm)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_metrics_comparison(axs, best_knn_and_svm_by_fold)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/model_comparison_{dataset_name}.png", dpi=300)
plt.show()
# %%
# Calculate means for each metric across folds
best_knn_and_svm["mean_f1"] = best_knn_and_svm[f1_cols].mean(axis=1)
best_knn_and_svm["mean_train_time"] = best_knn_and_svm[train_time_cols].mean(axis=1)
best_knn_and_svm["mean_test_time"] = best_knn_and_svm[test_time_cols].mean(axis=1)

# Now group by model and get means
best_knn_and_svm_summary = best_knn_and_svm.loc[
    :, ["model", "mean_f1", "mean_train_time", "mean_test_time"]
].copy()
best_knn_and_svm_summary.rename(
    columns=lambda x: x.replace("_", " ").title() + (" (s)" if "time" in x else ""), inplace=True
)
best_knn_and_svm_summary
write_latex_table(
    best_knn_and_svm_summary,
    f"{TABLES_DIR}/best_knn_and_svm_summary_{dataset_name}.tex",
    "Best KNN and SVM Models",
    precision=6,
)

# %%
metric_cols_map = {
    "Storage": storage_cols,
    "Training Time (s)": train_time_cols,
    "Testing Time (s)": test_time_cols,
    "F1 Score": f1_cols,
}
metric_pairs = [
    ("Storage", "Training Time (s)"),
    ("Storage", "Testing Time (s)"),
    ("Training Time (s)", "F1 Score"),
]
# %%
logging.info("Plotting KNN Reduction results")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_reduction_results_scatter(axes, metric_cols_map, metric_pairs, knn_reduction_results)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/KNN_reduction_distributions_{dataset_name}.png", dpi=300)
plt.show()
# %%
logging.info("Plotting SVM Reduction results")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_reduction_results_scatter(axes, metric_cols_map, metric_pairs, svm_reduction_results)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/SVM_reduction_distributions_{dataset_name}.png", dpi=300)
plt.show()
# %%
friedman_test(knn_reduction_results, train_time_cols)

# %%
expanded_knn_reduction_results = expand_data_per_fold(
    knn_reduction_results, "reduction_func", ["f1", "train_time", "test_time", "storage"]
)

fig = plot_independent_effects(
    expanded_knn_reduction_results,
    ["reduction_func"],
    y_cols=["f1", "train_time", "test_time", "storage"],
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/KNN_reduction_effects_{dataset_name}.png", dpi=300)
plt.show()
# %%
expanded_svm_reduction_results = expand_data_per_fold(
    svm_reduction_results, "reduction_func", ["f1", "train_time", "test_time", "storage"]
)

fig = plot_independent_effects(
    expanded_svm_reduction_results,
    ["reduction_func"],
    y_cols=["f1", "train_time", "test_time", "storage"],
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/SVM_reduction_effects_{dataset_name}.png", dpi=300)
plt.show()
# %%
friedman_test_df = pd.DataFrame(
    np.array(
        [
            [
                "KNN Top Sample of 8 F1 Scores",
                friedman_test(top_samples(knn_results, 8), f1_cols)[1],
            ],
            [
                "SVM Top Sample of 8 F1 Scores",
                friedman_test(top_samples(svm_results, 8), f1_cols)[1],
            ],
            [
                "KNN Linear Sample of 8 F1 Scores",
                friedman_test(linear_sample(knn_results, 8), f1_cols)[1],
            ],
            [
                "SVM Linear Sample of 8 F1 Scores",
                friedman_test(linear_sample(svm_results, 8), f1_cols)[1],
            ],
            [
                "KNN-Reduction Linear Sample of 8 F1 Scores",
                friedman_test(linear_sample(knn_reduction_results, 8), f1_cols)[1],
            ],
            [
                "SVM-Reduction Linear Sample of 8 F1 Scores",
                friedman_test(linear_sample(svm_reduction_results, 8), f1_cols)[1],
            ],
            [
                "KNN-Reduction Linear Sample of 8 Training Times",
                friedman_test(linear_sample(knn_reduction_results, 8), train_time_cols)[1],
            ],
            [
                "SVM-Reduction Linear Sample of 8 Training Times",
                friedman_test(linear_sample(svm_reduction_results, 8), train_time_cols)[1],
            ],
        ]
    ),
    columns=["name", "P Value"],
)
friedman_test_df.replace("nan", "1", inplace=True)
write_latex_table(
    friedman_test_df,
    f"{TABLES_DIR}/friedman_test_results_{dataset_name}.tex",
    f"Friedman Test Results {dataset_name}".title(),
    precision=8,
)
friedman_test_df
# %%
