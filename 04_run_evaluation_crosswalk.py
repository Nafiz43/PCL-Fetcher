"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from _constant_func import calculate_metrics, get_dataframes, calculate_metrics_for_crosswalk


crosswalk = pd.read_csv('data/cross_walk.csv')
original_report = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')

reports_to_process = -1
    # Parse arguments manually
if "--reports_to_process" in sys.argv:
    idx = sys.argv.index("--reports_to_process")
    reports_to_process = int(sys.argv[idx + 1])

print(f"Received value for reports_to_process: {reports_to_process}")




crosswalk.fillna(int(0), inplace=True)

crosswalk = crosswalk.drop(columns=['Accession Number'], errors='ignore')
crosswalk = crosswalk.drop(columns=['Modality'], errors='ignore')
crosswalk = crosswalk.drop(columns=['Completed'], errors='ignore')
crosswalk = crosswalk.drop(columns=['Exam Description'], errors='ignore')
crosswalk = crosswalk.drop(columns=['Report Text'], errors='ignore')
crosswalk = crosswalk.drop(columns=['Resident'], errors='ignore')

# print(crosswalk.columns)

original_report.fillna(int(0), inplace=True)

original_report = original_report.drop(columns=['Accession Number'],  errors='ignore')
original_report = original_report.drop(columns=['Modality'],  errors='ignore')
original_report = original_report.drop(columns=['Completed'],  errors='ignore')
original_report = original_report.drop(columns=['Completed REG'],  errors='ignore')
original_report = original_report.drop(columns=['Example Case List'],  errors='ignore')
original_report = original_report.drop(columns=['Exam Description'],  errors='ignore')
original_report = original_report.drop(columns=['Report Text'],  errors='ignore')
original_report = original_report.drop(columns=['Resident'],  errors='ignore')

# print(original_report.columns)
# print(len(original_report), len(crosswalk))

metrics_df = pd.DataFrame(columns=['Modality', 'TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'FPR', 'TPR'])

if(reports_to_process > 0):
    original_report = original_report.head(reports_to_process)

# original_report = original_report.head(100)
all_original_report_labels = []
all_crosswalk_labels = []

vascular_diagonsis_original_report_labels = []
vascular_diagonsis_crosswalk_labels = []

vascular_intervention_original_report_labels = []
vascular_intervention_crosswalk_labels =  []

non_vascular_intervention_original_report_labels = []
non_vascular_intervention_crosswalk_labels =  []

count = 1
for outer_index, outer_row in original_report.iterrows():  # Traversing original labels (master CSV)
    print("Report:", count)  # Debugging output
    count += 1
    original_report_exam_code = outer_row['Exam Code']
    original_report_labels = [int(x) for x in outer_row.iloc[1:].tolist()]

    all_original_report_labels.extend(original_report_labels)
    vascular_diagonsis_original_report_labels.extend(original_report_labels[0:8])
    vascular_intervention_original_report_labels.extend(original_report_labels[8:8+15])
    non_vascular_intervention_original_report_labels.extend(original_report_labels[8+15:8+15+16])
    print(original_report_labels, len(original_report_labels))
    
    k = 0
    for inner_index, inner_row in crosswalk.iterrows():  # Traversing crosswalk CSV
        if inner_row['Exam Code'] == original_report_exam_code:
            crosswalk_labels =  [int(x) for x in inner_row.iloc[1:].tolist()]

            all_crosswalk_labels.extend(crosswalk_labels)
            vascular_diagonsis_crosswalk_labels.extend(crosswalk_labels[0:8])
            vascular_intervention_crosswalk_labels.extend(crosswalk_labels[8:8+15])
            non_vascular_intervention_crosswalk_labels.extend(crosswalk_labels[8+15:8+15+16])
            break
    print(crosswalk_labels, len(crosswalk_labels))

metrics = calculate_metrics_for_crosswalk(all_original_report_labels, all_crosswalk_labels)
new_row = get_dataframes(metrics, "All")
metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)


metrics = calculate_metrics(vascular_diagonsis_original_report_labels, vascular_diagonsis_crosswalk_labels)
new_row = get_dataframes(metrics, "VascularDiagonsis")
metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

metrics = calculate_metrics(vascular_intervention_original_report_labels, vascular_intervention_crosswalk_labels)
new_row = get_dataframes(metrics, "VascularIntervention")

metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)



metrics = calculate_metrics(non_vascular_intervention_original_report_labels, non_vascular_intervention_crosswalk_labels)
new_row = get_dataframes(metrics, "NonVascularIntervention")

metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)


# Plot ROC Curves for Different Modalities


plt.figure(figsize=(8, 6))

for i, row in metrics_df.iterrows():
    fpr_values = (row["FPR"])  # Convert string representation of list to actual list
    tpr_values = (row["TPR"])  

    plt.plot(fpr_values, tpr_values, marker="o", linestyle="-", label=row["Modality"])

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves for Different Modalities")
plt.legend()
plt.grid()

plt.savefig('figs/04_crosswalk_roc_curves.png')

metrics_df = metrics_df.drop('FPR', axis=1)
metrics_df = metrics_df.drop('TPR', axis=1)

print(metrics_df)

# metrics_df = metrics_df.drop("FPR", axis=1)
# metrics_df = metrics_df.drop("TPR", axis=1)
metrics_df["Model-Name"] = "Cross-Walk"

metrics_df.to_csv("results/crosswalk_performance.csv", index=False)

print("CSV file saved as crosswalk_performance.csv")