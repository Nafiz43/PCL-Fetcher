"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
#This file collects question numbers from the fp and fn files

import re

import matplotlib.pyplot as plt
from collections import Counter


fn_case_file = "results/crosswalk_fn.txt"
fp_case_file = "results/crosswalk_fp.txt"


# fn_case_file = "results/CoT0anthropic.claude_FINAL.csv_false_negative_cases.txt"
# fp_case_file = "results/CoT0anthropic.claude_FINAL.csv_false_positive_cases.txt"

########## THE FOLLOWING IS FOR FN CASES ##########
with open(fn_case_file, "r") as file:
    text = file.read()

# Regular expression pattern to match "QUESTION-" followed by a number
pattern = r"QUESTION-(\d+)"

# Find all matches and store them in a list
question_numbers = re.findall(pattern, text)

# Convert to integers if needed
question_numbers = [int(num) for num in question_numbers]

print(question_numbers)


# Count occurrences of each unique number
counter = Counter(question_numbers)

# Ensure all numbers from 1 to 40 are present, even if they have zero occurrences
all_numbers = list(range(1, 41))  # Numbers from 1 to 40
occurrences = [counter.get(num, 0) for num in all_numbers]  # Get count or 0 if not present

# Plot bar chart
plt.figure(figsize=(10, 4))
plt.bar(all_numbers, occurrences, color='lightcoral')

# Labels and title
plt.xlabel("Number")
plt.ylabel("Occurrences")
plt.title("Queston Numbers in False Negative Cases")

# Set x-axis ticks to include all numbers from 1 to 40
plt.xticks(all_numbers, rotation=90)  # Rotate for better readability if needed
plt.savefig('figs/06_fn_question_numbers_barplot.png')

# Show plot
# plt.show()
########## THE ABOVE IS FOR FN CASES ##########



########## THE FOLLOWING IS FOR FP CASES ##########

with open(fp_case_file, "r") as file:
    text = file.read()

# Regular expression pattern to match "QUESTION-" followed by a number
pattern = r"QUESTION-(\d+)"

# Find all matches and store them in a list
question_numbers = re.findall(pattern, text)

# Convert to integers if needed
question_numbers = [int(num) for num in question_numbers]

print(question_numbers)


# Count occurrences of each unique number
counter = Counter(question_numbers)

# Ensure all numbers from 1 to 40 are present, even if they have zero occurrences
all_numbers = list(range(1, 40))  # Numbers from 1 to 40
occurrences = [counter.get(num, 0) for num in all_numbers]  # Get count or 0 if not present

# Plot bar chart
plt.figure(figsize=(10, 4))
plt.bar(all_numbers, occurrences, color='skyblue')

# Labels and title
plt.xlabel("Number")
plt.ylabel("Occurrences")
plt.title("Queston Numbers in False Positive Cases")

# Set x-axis ticks to include all numbers from 1 to 40
plt.xticks(all_numbers, rotation=90)  # Rotate for better readability if needed
plt.savefig('figs/06_fp_question_numbers_barplot.png')


print("result saved at: `figs/06_fp_question_numbers_barplot.png` and `figs/06_fn_question_numbers_barplot.png` files")

########## THE ABOVE IS FOR FP CASES ##########

from PIL import Image

# Load the saved images
img1 = Image.open("figs/06_fp_question_numbers_barplot.png")
img2 = Image.open("figs/06_fn_question_numbers_barplot.png")

# Create a figure with two subplots and reduce white space
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)  # Adjust size

# Show the images in the subplots
axes[0].imshow(img1)
axes[0].axis("off")  # Hide axes
axes[0].set_title("False Positives", fontsize=14)

axes[1].imshow(img2)
axes[1].axis("off")  # Hide axes
axes[1].set_title("False Negatives", fontsize=14)

# Save the combined figure with less padding
fig.savefig("figs/combined_barplots_tight.png", dpi=300, bbox_inches="tight")
print("Combined figure saved at: `figs/combined_barplots_tight.png` file")

# Show the combined figure
plt.show()





