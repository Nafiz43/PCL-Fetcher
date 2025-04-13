import pandas as pd
df = pd.read_csv('prompt-files/PCL_Questions_V5.csv')
saving_path1 = 'data/formatted_prompts_IP.tex'
with open(saving_path1, 'w', encoding='utf-8') as f:
    for i, row in enumerate(df['Questions'], start=1):
        latex_block = f"""\\texttt{{\\normalsize Prompt for Question  {i}:}}
\\begin{{lstlisting}}[]
{row}
\\end{{lstlisting}}\n\n"""
        f.write(latex_block)


df = pd.read_csv('prompt-files/PCL_Questions_V5_CoT.csv')
saving_path2='data/formatted_prompts_CoT.tex'
with open(saving_path2, 'w', encoding='utf-8') as f:
    for i, row in enumerate(df['Questions'], start=1):
        latex_block = f"""\\texttt{{\\normalsize Prompt for Question  {i}:}}
\\begin{{lstlisting}}[]
{row}
\\end{{lstlisting}}\n\n"""
        f.write(latex_block)

print(f'Formatted prompts saved in {saving_path1} and {saving_path2} files')