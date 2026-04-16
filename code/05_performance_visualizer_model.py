"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# It will be either input_file = 'performance_llm.csv' or input_file = 'performance_crosswalk.csv'

# input_file = 'performance_crosswalk.csv'

input_file = 'performance_llm.csv'

data = pd.read_csv(input_file)



# Create DataFrame
df = pd.DataFrame(data)

df = df.drop(columns=['TP', 'TN', 'FP', 'FN'])

# Set plot style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Melt DataFrame for easier plotting
df_melted = df.melt(id_vars=["Modality"], var_name="Metric", value_name="Score")

# Create barplot
sns.barplot(x="Modality", y="Score", hue="Metric", data=df_melted)


# plt.figure(figsize=(10, 5))

# Labels and title
plt.xlabel("Modality")
plt.ylabel("Score")
plt.title("Weighted Precision, Recall, and F1-Score by Modality")
plt.ylim(0.95, 1.0)  # Adjust y-axis for better visualization
plt.legend(title="Metrics")

# Rotate x-axis labels for better readability
plt.xticks(rotation=10)

input_file = input_file.replace('.csv', '_barplot.png')

plt.savefig(input_file)

# Show plot
plt.show()
