"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


df = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')
df= df.fillna(0)
print("Count of unique residents:", df['Resident'].nunique())
resident_counts = df['Resident'].value_counts().to_dict()

resident_names = list(resident_counts)
resident_reports_annotated = []
for resident in resident_counts:
    resident_reports_annotated.append(resident_counts[resident])
    print(resident_counts[resident])
resident_reports_annotated
print("average number of reports annotated per resident:", np.mean(resident_reports_annotated))




# Original data
counts = resident_reports_annotated
x = np.arange(1, len(counts) + 1)

# Create smooth x-values
x_smooth = np.linspace(x.min(), x.max(), 300)

# Fit a spline and evaluate smooth y-values
spline = make_interp_spline(x, counts, k=3)  # k=3 for cubic spline
y_smooth = spline(x_smooth)

# Plot
plt.figure(figsize=(8, 5))

# Bar plot
# bars = plt.bar(x, counts, alpha=0.7)
bars = plt.bar(x, counts, color='lightblue', alpha=0.7)


# Smooth curve
plt.plot(x_smooth, y_smooth, color='blue', linewidth=1, label='Smoothed Trend')

# Add labels inside bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height/2, f'{int(height)}',
             ha='center', va='center', color='black', fontsize=10, fontweight='bold')

plt.xlabel('Resident Index')
plt.ylabel('Report Count')
# plt.title('Procedures Performed by Residents (Smoothed Trend)')
plt.xticks(x)
# plt.legend()
plt.tight_layout()
plt.savefig('figs/resident_report_distribution.png', dpi=300)
plt.show()
