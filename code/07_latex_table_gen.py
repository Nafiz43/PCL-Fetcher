"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
import numpy as np
import os
import sys


crosswalk = pd.read_csv('results/crosswalk_performance.csv')
crosswalk = crosswalk.rename(columns={'Unnamed: 0': 'model'})
crosswalk.drop(columns=['Sensitivity-Weighted', 'Specificity-Weighted','Precision', 'Precision-Weighted', 'F1-Score-Weighted', 'Model-Name'], inplace=True)
crosswalk.insert(loc=0, column='Prompting Method', value="NA")
crosswalk.insert(loc=0, column='Model-Name', value="Cross-Walk")
crosswalk.insert(loc=0, column='Model Type', value="Benchmark")
crosswalk

df = pd.read_csv('results/all_models.csv')
df.drop(columns=['Sensitivity-Weighted', 'Specificity-Weighted', 'Precision-Weighted', 'F1-Score-Weighted', 'Precision'], inplace=True)
df['Modality']  = df['Modality'].str.replace(r"[\[\],']", '', regex=True)
df


best_local_cot = df[df['Model-Name'] == 'CoT0qwen2.5_72b-FINA']
best_local_ip = df[df['Model-Name'] == 'IPO0qwen2.5_72b-FINA']

# best_cm_cot 
best_cm_ip  = df[df['Model-Name'] == 'IPO0anthropic.claude']
best_cm_cot = df[df['Model-Name'] == 'CoT0anthropic.claude']


best_cm_ip.insert(loc=0, column='Prompting Method', value="IP")
best_cm_ip.drop(columns=['Model-Name'], inplace=True)

best_cm_ip.insert(loc=0, column='Model-Name', value="Claude-3.5-Haiku")
best_cm_ip.insert(loc=0, column='Model Type', value="Commercial")

best_cm_ip


best_cm_cot.insert(loc=0, column='Prompting Method', value="CoT")
best_cm_cot.drop(columns=['Model-Name'], inplace=True)
best_cm_cot.insert(loc=0, column='Model-Name', value="Claude-3.5-Haiku")
best_cm_cot.insert(loc=0, column='Model Type', value="Commercial")
best_cm_cot


best_local_cot.insert(loc=0, column='Prompting Method', value="CoT")
best_local_cot.drop(columns=['Model-Name'], inplace=True)
best_local_cot.insert(loc=0, column='Model-Name', value="Qwen-2.5:72B")
best_local_cot.insert(loc=0, column='Model Type', value="Local")
best_local_cot


best_local_ip.insert(loc=0, column='Prompting Method', value="IP")
best_local_ip.drop(columns=['Model-Name'], inplace=True)
best_local_ip.insert(loc=0, column='Model-Name', value="Qwen-2.5:72B")
best_local_ip.insert(loc=0, column='Model Type', value="Local")
best_local_ip


combined_df = pd.concat([crosswalk, best_local_ip, best_local_cot, best_cm_ip, best_cm_cot], axis=0, ignore_index=True)
print("main performance table of the paper")
# combined_df.drop(columns=['Unnamed: 0'], inplace=True)
combined_df


import pandas as pd
from _stat_gen import *

# Specify which columns to apply multirow to
cols_to_merge = ['Model Type', 'Model-Name', 'Prompting Method']
df_multirow = apply_multirow(combined_df, cols_to_merge)

latex_code = df_multirow.to_latex(
    index=False,
    caption="Performance Comparison of Models Across Modalities",
    label="tab:model-performance",
    column_format='|l|l|l|l|r|r|r|r|r|r|r|',
    escape=False,
    float_format="%.2f"  # <<< forces rounding to 2 decimal places
)

# Add required package manually in your LaTeX preamble:
# \usepackage{multirow}

print("LaTeX code for the main table:")
print(latex_code)
main_table_saving_path = 'paper-tables/main.tex'

os.makedirs('paper-tables', exist_ok=True)

with open(main_table_saving_path, "w") as f:
    f.write(latex_code)



model_order = {
    'medllama2': 0,
    'mixtral': 1,
    'llama3.3': 2,
    'llama3-med-42': 3
}

# Step 2: Extract model family from 'Model-Name'
def extract_family(name):
    name = name.lower()
    if 'medllama2' in name:
        return 'medllama2'
    elif 'mixtral' in name:
        return 'mixtral'
    elif 'llama3.3' in name:
        return 'llama3.3'
    else:
        return 'llama3-med-42'

def sort_dataframe(df):
        
    df['model_family'] = df['Model-Name'].apply(extract_family)
    df.loc[df['model_family'] == 'medllama2', 'Model-Name'] = "Medllama2:7B"
    df.loc[df['model_family'] == 'mixtral', 'Model-Name'] = "Mixtral:8x7B"
    df.loc[df['model_family'] == 'llama3.3', 'Model-Name'] = "Llama3.3:70B"
    df.loc[df['model_family'] == 'llama3-med-42', 'Model-Name'] = "Llama3-Med42:70B"

    # print(df['model_family'])
    # Step 3: Add sort key and sort
    df['sort_key'] = df['model_family'].map(model_order)
    sorted_df = df.sort_values(by='sort_key').drop(columns=['sort_key'])

    # Optional: move 'model_family' to the end or drop it
    sorted_df = sorted_df[[col for col in sorted_df.columns if col != 'model_family'] + ['model_family']]
    return sorted_df


df1 = df[~df['Model-Name'].str.contains('qwen2.5|claude|med42-8b|lama3_8b', case=False, na=False)]
df1
ip_content = df1[df1['Model-Name'].str.contains('IPO', case=False, na=False)]
ip_content.insert(loc=0, column='Prompting Method', value="IP")

cot_content = df1[df1['Model-Name'].str.contains('CoT', case=False, na=False)]
cot_content.insert(loc=0, column='Prompting Method', value="CoT")

appendix_df = pd.concat([ip_content, cot_content], axis=0, ignore_index=True)

appendix_df= sort_dataframe(appendix_df)
appendix_df.drop(columns=['model_family'], inplace=True)
# appendix_df.insert(loc=0, column='Prompting Method', value="IP")
appendix_df.insert(loc=0, column='Model-Namex', value=appendix_df['Model-Name'])


appendix_df.drop(columns=['Model-Name'], inplace=True)
# appendix_df.insert(loc=0, column='Model Type', value="Local")

# Step 2: Define custom sort order for 'Modality'
modality_order = ['All', 'VascularDiagnosis', 'VascularIntervention', 'NonVascularIntervention']

# Step 3: Convert 'Modality' to categorical with defined order
appendix_df['Modality'] = pd.Categorical(appendix_df['Modality'], categories=modality_order, ordered=True)

# Step 4: Group by and sort
appendix_df = appendix_df.groupby('Model-Namex').apply(
    lambda group: group.sort_values(['Prompting Method', 'Modality'], ascending=[False, True])
).reset_index(drop=True)

appendix_df
cols_to_merge = ['Model-Namex']
appendix_df = apply_multirow(appendix_df, cols_to_merge)

latex_code = appendix_df.to_latex(
    index=False,
    # caption="Performance Comparison of Models Across Modalities",
    # label="tab:model-performance",
    # column_format='|l|l|l|l|r|r|r|r|r|r|r|',
    escape=False,
    float_format="%.2f"  # <<< forces rounding to 2 decimal places
)

# Add required package manually in your LaTeX preamble:
# \usepackage{multirow}


print("LaTeX code for the appendix table:")
print(latex_code)

appendix_table_saving_path = 'paper-tables/appendix.tex'

with open(appendix_table_saving_path, "w") as f:
    f.write(latex_code)

print(f"Fles saved in {main_table_saving_path} and {appendix_table_saving_path}")