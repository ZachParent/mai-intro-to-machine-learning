import os

def format_column_names(df):
    result = df.copy()
    result.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
    return result


def write_latex_table(df, filename, caption, precision=3, longtable=False):
    label = os.path.splitext(os.path.basename(filename))[0]
    if longtable:
        # Create the table manually instead of using pandas styling
        latex_table = (
            "\\begin{longtable}{" + "|".join(["c"] * len(df.columns)) + "}\n"
            "\\caption{" + caption + "} \\label{tab:" + label + "} \\\\\n"
            "\\hline\n"
            # Column headers
            f"{' & '.join(df.columns)} \\\\\n"
            "\\hline\n"
            "\\endfirsthead\n\n"
            # Continuation header
            f"\\multicolumn{{{len(df.columns)}}}{{c}}{{ {caption} -- Continued}} \\\\\n"
            "\\hline\n"
            f"{' & '.join(df.columns)} \\\\\n"
            "\\hline\n"
            "\\endhead\n\n"
            # Footer
            "\\hline\n"
            f"\\multicolumn{{{len(df.columns)}}}{{r}}{{Continued on next page}}\n"
            "\\endfoot\n\n"
            "\\hline\n"
            "\\endlastfoot\n"
        )
        
        # Add table content
        for _, row in df.iterrows():
            latex_table += " & ".join([str(round(x, precision)) if isinstance(x, float) else str(x) for x in row]) + " \\\\\n"
        
        latex_table += "\\end{longtable}"
    else:
        # Use pandas styling for regular tables
        s = df.style
        s.table_styles = []
        s.caption = caption
        s.format(precision=precision)
        s.hide(level=0, axis=0)
        latex_table = s.to_latex(
            position_float="centering",
            multicol_align="|c|",
            hrules=True,
            label=f"tab:{label}"
        )

    with open(filename, "w") as f:
        f.write(latex_table)

# %%
def write_latex_table_summary(df, columns, filename, caption, sort_by="f1"):
    df = (
        df.sort_values(by=sort_by, ascending=False)
        .reset_index(drop=True)
        .assign(**{"": lambda x: x.index + 1})
        .loc[:, [""] + columns]
        .head(10)
        .rename(columns=lambda x: x.replace("_", " "))
    )
    write_latex_table(df, filename, caption)