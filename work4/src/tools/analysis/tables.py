import os
import pandas as pd


def generate_best_models_table(metrics_df: pd.DataFrame, output_path: str):
    """Generate a LaTeX table showing the best performing models for each dataset based on f_measure."""

    # Select relevant columns
    columns = [
        "dataset",
        "clustering_model",
        "reduction_method",
        "f_measure",
        "ari",
        "chi",
        "dbi",
        "clustering_runtime",
        "reduction_runtime",
    ]

    # Get best model for each dataset
    best_models = (
        metrics_df.sort_values("f_measure", ascending=False)
        .groupby("dataset")
        .first()
        .reset_index()
        .loc[:, columns]
    )

    # Format the table
    best_models = format_column_names(best_models)

    # Write to LaTeX
    caption = "Best Performing Models by Dataset (Based on F-Measure)"
    write_latex_table(best_models, output_path, caption, precision=4)


def generate_best_reduction_method_table(metrics_df: pd.DataFrame, output_path: str):
    """Generate a LaTeX table showing the best performing models for each reduction method based on f_measure."""

    # Select relevant columns
    columns = [
        "dataset",
        "clustering_model",
        "reduction_method",
        "f_measure",
        "ari",
        "chi",
        "dbi",
        "clustering_runtime",
        "reduction_runtime",
    ]

    # Get best model for each dataset
    best_models = (
        metrics_df.sort_values("f_measure", ascending=False)
        .groupby("reduction_method")
        .first()
        .reset_index()
        .loc[:, columns]
    )

    # Format the table
    best_models = format_column_names(best_models)

    # Write to LaTeX
    caption = "Best Performing Models by Reduction Method (Based on F-Measure)"
    write_latex_table(best_models, output_path, caption, precision=4)


def generate_top_models_by_dataset(
    metrics_df: pd.DataFrame, dataset_name: str, output_path: str
):
    """Generate a LaTeX table showing the top 10 configurations for a specific dataset."""
    # Filter for the dataset and select relevant columns
    columns = [
        "clustering_model",
        "reduction_method",
        "f_measure",
        "ari",
        "chi",
        "dbi",
        "clustering_runtime",
        "reduction_runtime",
    ]
    dataset_results = metrics_df[metrics_df["dataset"] == dataset_name]

    # Write summary table
    caption = f"Top 10 Configurations for {dataset_name.title()} Dataset"
    write_latex_table_summary(dataset_results, columns, output_path, caption)

def generate_top_models_by_reduction_method(
    metrics_df: pd.DataFrame, reduction_method: str, output_path: str
):
    """Generate a LaTeX table showing the top 10 configurations for a specific reduction method."""
    # Filter for the dataset and select relevant columns
    columns = [
        "clustering_model",
        "reduction_method",
        "f_measure",
        "ari",
        "chi",
        "dbi",
        "clustering_runtime",
        "reduction_runtime",
    ]
    dataset_results = metrics_df[metrics_df["reduction_method"] == reduction_method]

    # Write summary table
    caption = f"Top 10 Configurations for {reduction_method.title()}"
    write_latex_table_summary(dataset_results, columns, output_path, caption)


def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Format column names for LaTeX tables.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with formatted column names
    """
    result = df.copy()
    result.columns = [
        col.replace("_", " ").title() + (" (s)" if "runtime" in col.lower() else "")
        for col in result.columns
    ]
    return result


def write_latex_table(
    df: pd.DataFrame, filename: str, caption: str, precision: int = 3
):
    """Write DataFrame to a LaTeX table.

    Args:
        df: Input DataFrame
        filename: Output path for the LaTeX file
        caption: Table caption
        precision: Number of decimal places for float values
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Get label from filename
    label = os.path.splitext(os.path.basename(filename))[0]

    # Style the table
    styled_df = df.style
    styled_df.format(precision=precision)
    styled_df.hide(axis=0)  # Hide index

    with open(filename, "w") as f:
        latex = styled_df.to_latex()
        latex = latex.replace("_", "\\_")
        latex = latex.replace(
            "\\begin{tabular}",
            f"\\begin{{table*}}[ht!]\n\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\begin{{tabular}}",
        )
        latex = latex.replace("\\end{tabular}", "\\end{tabular}\n\\end{table*}")

        # Add horizontal line after header
        header_end = latex.find("\\\\", latex.find("\\begin{tabular}"))
        latex = latex[: header_end + 2] + "\\midrule\n" + latex[header_end + 2 :]

        f.write(latex)


def write_latex_table_summary(
    df: pd.DataFrame, columns: list, filename: str, caption: str, sort_by="f_measure"
):
    """Write a summary table showing top 10 entries sorted by a specific column.

    Args:
        df: Input DataFrame
        columns: List of columns to include
        filename: Output path for the LaTeX file
        caption: Table caption
        sort_by: Column to sort by (default: "f_measure")
    """
    df = (
        df.sort_values(by=sort_by, ascending=False)
        .reset_index(drop=True)
        .assign(**{"": lambda x: x.index + 1})
        .loc[:, [""] + columns]
        .head(10)
        .rename(columns=lambda x: x.replace("_", " "))
    )
    df = format_column_names(df)
    write_latex_table(df, filename, caption, precision=6)


def generate_model_best_configs_table(
    metrics_df: pd.DataFrame, model_name: str, output_path_base: str
):
    """Generate LaTeX tables showing the best configuration for each dataset for a specific model.
    Generates two tables:
    1. Best configurations (parameters)
    2. Performance metrics for those configurations

    Args:
        metrics_df: Input DataFrame with all metrics
        model_name: Name of the model to analyze
        output_path_base: Base path to save the LaTeX tables (will append _config and _performance)
    """
    # Filter for the specific model
    model_results = metrics_df[metrics_df["clustering_model"] == model_name]

    # Get parameter columns for this model
    param_cols = [
        col
        for col in model_results.columns
        if col
        not in [
            "dataset",
            "clustering_model",
            "reduction_method",
            "f_measure",
            "ari",
            "chi",
            "dbi",
            "clustering_runtime",
            "reduction_runtime",
        ]
        and not model_results[col].isna().all()
    ]

    # Get best configuration for each dataset
    best_configs = (
        model_results.sort_values("f_measure", ascending=False)
        .groupby("dataset")
        .first()
        .reset_index()
    )

    # Generate configurations table
    config_columns = ["dataset"] + param_cols
    config_df = best_configs.loc[:, config_columns]
    config_df = format_column_names(config_df)

    config_path = output_path_base.replace(".tex", "_config.tex")
    config_caption = f"Best Parameter Configurations for {model_name.replace('_', ' ').title()} by Dataset"
    write_latex_table(config_df, config_path, config_caption, precision=5)

    # Generate performance metrics table
    metrics_columns = [
        "dataset",
        "f_measure",
        "ari",
        "chi",
        "dbi",
        "clustering_runtime",
        "reduction_runtime",
    ]
    metrics_df = best_configs.loc[:, metrics_columns]
    metrics_df = format_column_names(metrics_df)

    metrics_path = output_path_base.replace(".tex", "_performance.tex")
    metrics_caption = f"Performance Metrics for Best {model_name.replace('_', ' ').title()} Configurations by Dataset"
    write_latex_table(metrics_df, metrics_path, metrics_caption, precision=4)
