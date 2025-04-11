# This script collects question numbers from FN and FP files and generates annotated bar plots

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def extract_question_counts(filepath, pattern=r"QUESTION-(\d+)", total_questions=39):
    with open(filepath, "r") as file:
        text = file.read()
    numbers = [int(num) for num in re.findall(pattern, text)]
    counter = Counter(numbers)
    all_numbers = list(range(1, total_questions + 1))
    occurrences = [counter.get(num, 0) for num in all_numbers]
    return all_numbers, occurrences

fn_case_file_crosswalk = "results/crosswalk_fn.txt"
fp_case_file_crosswalk = "results/crosswalk_fp.txt"


fn_case_file_cm = "results/CoT0anthropic.claude_FINAL.csv_false_negative_cases.txt"
fp_case_file_cm = "results/CoT0anthropic.claude_FINAL.csv_false_positive_cases.txt"


fn_case_file_local = "results/CoT0qwen2.5_72b-FINAL.csv_false_negative_cases.txt"
fp_case_file_local = "results/CoT0qwen2.5_72b-FINAL.csv_false_positive_cases.txt"


all_numbers, occurrences_fn_crosswalk = extract_question_counts(fn_case_file_crosswalk)
all_numbers, occurrences_fp_crosswalk = extract_question_counts(fp_case_file_crosswalk)

all_numbers, occurrences_fn_cm = extract_question_counts(fn_case_file_cm)
all_numbers, occurrences_fp_cm = extract_question_counts(fp_case_file_cm)

all_numbers, occurrences_fn_local = extract_question_counts(fn_case_file_local)
all_numbers, occurrences_fp_local = extract_question_counts(fp_case_file_local)


all_numbers.append("Total")
occurrences_fp_crosswalk.append(np.sum(occurrences_fp_crosswalk))
occurrences_fn_crosswalk.append(np.sum(occurrences_fn_crosswalk))
occurrences_fp_cm.append(np.sum(occurrences_fp_cm))
occurrences_fn_cm.append(np.sum(occurrences_fn_cm))
occurrences_fp_local.append(np.sum(occurrences_fp_local))
occurrences_fn_local.append(np.sum(occurrences_fn_local))



df = pd.DataFrame({
    'Question Number': all_numbers,
    'Crosswalk - FP': occurrences_fp_crosswalk,
    'Crosswalk - FN': occurrences_fn_crosswalk,
    'Claude - FP': occurrences_fp_cm,
    'Claude - FN': occurrences_fn_cm,
    'Qwen - FP': occurrences_fp_local,
    'Qwen - FN': occurrences_fn_local
})

df = df.to_latex(index=False, escape=False,  float_format="%.2f")
print(df)


