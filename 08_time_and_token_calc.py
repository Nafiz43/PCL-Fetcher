"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
import numpy as np

from _stat_gen import *



def get_prompt_method(method: str) -> str:
    if "CoT" in method:
        return "CoT"
    elif "IP" in method or "IPO" in method:
        return "IP"
    else:
        return "None"

def get_model_type(model: str) -> str:
    if "claude" in model:
        return "Commercial"
    else:
        return "Local"
    
import os
local_history_directory = 'local_chat_history'
csv_files = [f for f in os.listdir(local_history_directory) if f.endswith('.csv')]
csv_files

ag_df = pd.DataFrame()
for csv in csv_files:
    print(f"Processing {csv}...")
    df = pd.read_csv('local_chat_history/'+csv)
    # df = df.tail(3900)  # Limit to the first 1000 rows
    df['token_count'] = df['reason'].apply(lambda x: count_tokens(x) if pd.notnull(x) else 0)

    df.drop(columns=['accession_number', 'question', 'answer', 'reason', 'model_name'], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df =df.head(975)
    

    df["Time_Diff_Seconds"] = df["timestamp"].diff().dt.total_seconds()

    # Drop the first row as it has NaN time diff
    time_diffs = df["Time_Diff_Seconds"].dropna()

    token_calc = df['token_count'].dropna()

    combined_stats = {
    "Model Name": csv.replace('.csv', ''),
    'Model Type': get_model_type(csv),
    "Prompt Method": get_prompt_method(csv),
    "Max Time Diff": time_diffs.max(),
    "Min Time Diff": time_diffs.min(),
    "Mean Time Diff": round(time_diffs.mean(), 2),
    "Std Dev Time Diff": round(time_diffs.std(), 2),
    "Max Tokens": token_calc.max(),
    "Min Tokens": token_calc.min(),
    "Mean Tokens": round(token_calc.mean(), 2),
    "Std Dev Tokens": round(token_calc.std(), 2),
    }

    # Convert to DataFrame
    combined_stats_df = pd.DataFrame([combined_stats])
    ag_df = pd.concat([ag_df, combined_stats_df], ignore_index=True)
    # print(combined_stats_df)


ag_df = ag_df.sort_values("Prompt Method", ascending=False)
ag_df


winner_models = "qwen|claude"
main_paper_table_df = ag_df[ag_df['Model Name'].str.contains(winner_models)]
main_paper_table_df = main_paper_table_df.sort_values("Model Type", ascending=False)
main_paper_table_df.to_csv('paper-tables/main_paper_table.csv', index=False)

# cols_to_merge = ['Model Name', 'Model Type']
# main_paper_table_df = apply_multirow(main_paper_table_df, cols_to_merge)

main_paper_table_df_latex = main_paper_table_df.to_latex(index=False, escape=False,  float_format="%.2f")
print(main_paper_table_df_latex)
main_paper_table_df

main_paper_table_df_cost_and_token = 'paper-tables/main_cost.tex'

with open(main_paper_table_df_cost_and_token, "w") as f:
    f.write(main_paper_table_df_latex)


appendix_df = ag_df[~ag_df['Model Name'].str.contains(winner_models)]
appendix_df = appendix_df.drop(columns=['Model Type'])

appendix_df = appendix_df.groupby('Model Name').apply(
    lambda group: group.sort_values('Prompt Method', ascending=False)
).reset_index(drop=True)

cols_to_merge = ['Model Name']
appendix_df = apply_multirow(appendix_df, cols_to_merge)


appendix_df_latex = appendix_df.to_latex(index=False, escape=False, float_format="%.2f" )
print(appendix_df_latex)
appendix_df


appendix_paper_table_df_cost_and_token = 'paper-tables/appendix_cost.tex'

with open(appendix_paper_table_df_cost_and_token, "w") as f:
    f.write(appendix_df_latex)
