# This script collects question numbers from FN and FP files and generates annotated bar plots

import re
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

def plot_bar_chart(all_numbers, occurrences, color, title, output_path):
    plt.figure(figsize=(10, 4))
    bars = plt.bar(all_numbers, occurrences, color=color)

    # Add count labels just inside the top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.5,  # slightly below the top
                f'{int(height)}',
                ha='center',
                va='top',
                fontsize=8,
                color='black'  # white for tall bars, black for short
            )

    plt.xlabel("Question Number")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.xticks(all_numbers, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# File paths
fn_case_file = "results/crosswalk_fn.txt"
fp_case_file = "results/crosswalk_fp.txt"

# fn_case_file = "results/CoT0anthropic.claude_FINAL.csv_false_negative_cases.txt"
# fp_case_file = "results/CoT0anthropic.claude_FINAL.csv_false_positive_cases.txt"


fp_saving_path = "figs/06_fp_question_numbers_barplot_crosswalk.png"
fn_saving_path = "figs/06_fn_question_numbers_barplot_crosswalk.png"

# Extract and plot FN cases
all_numbers_fn, occurrences_fn = extract_question_counts(fn_case_file)
plot_bar_chart(
    all_numbers_fn, occurrences_fn, color='lightcoral',
    title="",
    output_path=fn_saving_path
)

# Extract and plot FP cases
all_numbers_fp, occurrences_fp = extract_question_counts(fp_case_file)
plot_bar_chart(
    all_numbers_fp, occurrences_fp, color='skyblue',
    title="",
    output_path=fp_saving_path
)

print("Plots saved: `{fn_saving_path}` and `{fp_saving_path}`")

# Load the saved images
img_fn = Image.open(fn_saving_path)
img_fp = Image.open(fp_saving_path)

# Combine plots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axes[0].imshow(img_fp)
axes[0].axis("off")
axes[0].set_title("False Positives", fontsize=14)

axes[1].imshow(img_fn)
axes[1].axis("off")
axes[1].set_title("False Negatives", fontsize=14)

# Save and display
fig.savefig("figs/combined_barplots_tight.png", dpi=300, bbox_inches="tight")
print("Combined figure saved at: `figs/combined_barplots_tight.png`")

plt.show()
