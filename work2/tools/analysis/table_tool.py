import os

def format_column_names(df):
    df.rename(columns=lambda x: x.replace('_', ' ').title(), inplace=True)
    return df

def write_latex_table(df, filename, caption, precision=3):
    label = os.path.splitext(os.path.basename(filename))[0]
    s = df.style
    s.table_styles = []
    s.caption = caption
    s.format(
        precision=precision,
    )
    s.hide(level=0, axis=0)
    latex_table = s.to_latex(position_float="centering",
                             multicol_align="|c|",
                             hrules=True,
                             label=f"tab:{label}",
                             )
    with open(filename, "w") as f:
        f.write(latex_table)