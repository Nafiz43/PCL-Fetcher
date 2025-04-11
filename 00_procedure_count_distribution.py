import pandas as pd
import numpy as np


df = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')
df= df.fillna(0)

df.drop(['Accession Number', 'Modality', 'Exam Code', 'Completed',
       'Completed REG', 'Example Case List', 'Exam Description', 'Report Text', 'Resident'], axis=1, inplace=True)

distribution_count = []
for i in range(0, 40):
    distribution_count.append(0)
print(distribution_count)
procedure_count =[]

for i in range(len(df)):
    row = df.iloc[i]
    cnt = 0
    for value in row:
        if value == 1:
            cnt += 1
    # print(cnt)
    distribution_count[cnt]= distribution_count[cnt] + 1
print(distribution_count)
np.sum(distribution_count)


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Your data
# distribution_count = [220, 292, 176, 74, 14, 2, 6] + [0]*33
bins = list(range(len(distribution_count)))

# Create smooth x and y for curve using interpolation
x = np.array(bins)
y = np.array(distribution_count)
x_smooth = np.linspace(x.min(), x.max(), 500)
spline = make_interp_spline(x, y, k=3)  # cubic spline
y_smooth = spline(x_smooth)

# Plot bar chart and curve
plt.figure(figsize=(12, 6))
bars = plt.bar(bins, distribution_count, color='yellowgreen', edgecolor='black', alpha=0.6, label='Histogram')
plt.plot(x_smooth, y_smooth, color='gray', linewidth=2.5, label='Smoothed Curve')

# Add bold text labels above bars
for bin_val, count in zip(bins, distribution_count):
    if count > 0:
        plt.text(
            bin_val, count + 3, str(count),
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )

plt.xticks(ticks=bins)
plt.xlabel('Identified Procedures')
plt.ylabel('Report Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.legend()
plt.tight_layout()
plt.savefig('figs/procedure_count_distribution.png', dpi=300)

plt.show()