import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# It will be either input_file = 'performance_llm.csv' or input_file = 'performance_crosswalk.csv'

# input_file = 'performance_crosswalk.csv'

input_file_llm = 'results/all_models.csv'
input_file_crosswalk = 'crosswalk_performance.csv'




### LLM Performance Vissualization ####
data = pd.read_csv(input_file_llm)

df = pd.DataFrame(data)

df = df.drop(columns=['TP', 'TN', 'FP', 'FN', 'Sensitivity-Weighted','Specificity-Weighted','Precision-Weighted','F1-Score-Weighted','Model-Name'])

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

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

input_file_llm = input_file_llm.replace('.csv', '_barplot.png')

plt.savefig('figs/05_'+input_file_llm.replace('results/', '').replace('.csv', ''))
### LLM Performance Vissualization ####





### Crosswalk Performance Vissualization ####
data = pd.read_csv(input_file_crosswalk)
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

input_file_crosswalk = input_file_crosswalk.replace('.csv', '_barplot.png')

plt.savefig('figs/05_'+input_file_crosswalk)
### Crosswalk Performance Vissualization ####



# Show plot
# plt.show()
