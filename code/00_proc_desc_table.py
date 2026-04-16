"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
import os
from _stat_gen import apply_multirow

df = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')
df.drop(['Accession Number', 'Modality', 'Exam Code', 'Completed',
       'Completed REG', 'Example Case List', 'Exam Description', 'Report Text', 'Resident'], axis=1, inplace=True)

procedures = df.columns

df = pd.DataFrame()

df['# of Procedure'] = [i for i in range(1, 40)]
df['Procedure'] = procedures

modality = ['Vascular Diagonosis'] * 8 + \
           ['Vascular Intervention'] * 15 + \
           ['Non Vascular Intervention'] * 16

# Assign the modality column
df['Modality'] = modality

df
cols_to_merge = ['Modality']
df_multirow = apply_multirow(df, cols_to_merge)

latex_code = df_multirow.to_latex(
    index=False,
    escape=False,
)
print(latex_code)

output_dir = 'paper-tables'
proc_table_saving_path = os.path.join(output_dir, 'proc_table.tex')

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)
6
with open(proc_table_saving_path, "w") as f:
    f.write(latex_code)

print(f"procedure table saved to {proc_table_saving_path}")