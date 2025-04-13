"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""

import tiktoken
from copy import deepcopy

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a given text using the tokenizer for a specific model.

    Args:
        text (str): The input text to tokenize.
        model (str): The OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4").

    Returns:
        int: Number of tokens in the input text.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def apply_multirow(df, multirow_columns):
    df = deepcopy(df)
    for col in multirow_columns:
        prev_val = None
        count = 0

        # First pass to count how many repeated rows per value
        value_counts = []
        for val in df[col]:
            if val == prev_val:
                count += 1
            else:
                if prev_val is not None:
                    value_counts.append((prev_val, count))
                prev_val = val
                count = 1
        value_counts.append((prev_val, count))

        # Second pass to replace with \multirow and blanks
        i = 0
        for val, span in value_counts:
            if span == 1:
                i += 1
                continue
            df.at[i, col] = f"\\multirow{{{span}}}{{*}}{{{val}}}"
            for j in range(1, span):
                df.at[i + j, col] = ''
            i += span
    return df

replacement_dict = {
    'IPO0medllama2-FINAL': 'Medllama2:7B',
    'IPO0qwen2.5_72b-FINAL': 'Qwen2.5:70B',
    'IPO0llama3.3_70b-FINAL': 'Llama3:70B',
    'IPO0anthropic.claude-3-5-haiku_FINAL': 'Claude-3.5-Haiku',
    'IPO0mixtral_8x7b-instruct-FINAL': 'Mixtral:8x7B',
    'CoT0qwen2.5_72b-FINAL': 'Qwen2.5:70B',
    'CoT0llama3.3_70bFinal': 'Llama3:70B',
    'CoT0anthropic.claude_FINAL': 'Claude-3.5-Haiku',
    'CoT0mixtral_8x7b-FINAL': 'Mixtral:8x7B',
    'CoT0medllama2_latest-12025-03-26 02_25': 'Medllama2:7B',
    'IPO0llama3-med42-70b-12025-04-06 03_26': 'Llama3-Med42:70B',
    'CoT0llama3-med42-70b-12025-04-07 22_08': 'Llama3-Med42:70B'
}