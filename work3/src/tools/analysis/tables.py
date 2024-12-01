import os
import pandas as pd

def generate_best_models_table(metrics_df: pd.DataFrame, output_path: str):
    """Generate a LaTeX table showing the best performing models for each dataset based on f_measure."""
    
    # Select relevant columns
    columns = ['dataset', 'model', 'f_measure', 'ari', 'chi', 'dbi', 'runtime']
    
    # Get best model for each dataset
    best_models = (metrics_df
                  .sort_values('f_measure', ascending=False)
                  .groupby('dataset')
                  .first()
                  .reset_index()
                  .loc[:, columns])
    
    # Format the table
    best_models = format_column_names(best_models)
    
    # Write to LaTeX
    caption = "Best Performing Models by Dataset (Based on F-Measure)"
    write_latex_table(best_models, output_path, caption, precision=4)

def generate_top_models_by_dataset(metrics_df: pd.DataFrame, dataset_name: str, output_path: str):
    """Generate a LaTeX table showing the top 10 configurations for a specific dataset."""
    
    # Filter for the dataset and select relevant columns
    columns = ['model', 'f_measure', 'ari', 'chi', 'dbi', 'runtime']
    dataset_results = metrics_df[metrics_df['dataset'] == dataset_name]
    
    # Write summary table
    caption = f"Top 10 Configurations for {dataset_name.title()} Dataset"
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

def write_latex_table(df: pd.DataFrame, filename: str, caption: str, precision: int = 3):
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
    
    # Write to LaTeX
    with open(filename, 'w') as f:
        f.write(styled_df.to_latex())

def write_latex_table_summary(df: pd.DataFrame, columns: list, filename: str, caption: str, sort_by="f_measure"):
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
