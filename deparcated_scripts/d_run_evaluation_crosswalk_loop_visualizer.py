import matplotlib.pyplot as plt
import pandas as pd

metrics_df = pd.read_csv("crosswalk_performance.csv")

# First figure (Precision, Recall, F1-Score)
plt.figure(figsize=(10, 5))
for column in ["Weighted-Precision", "Weighted-Recall", "Weighted-F1-Score"]:
    plt.plot(metrics_df["report_count"], metrics_df[column], marker="o", label=column)

plt.xlabel("Report Count")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score vs Report Count")
# plt.xscale("log")  # Log scale for better visualization
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("crosswalk_loop_metrics.png")
plt.show()

# Second figure (TP, TN, FP, FN)
plt.figure(figsize=(10, 5))
for column in ["FP", "FN"]:
    plt.plot(metrics_df["report_count"], metrics_df[column], marker="o", label=column)

print("hello")
plt.xlabel("Report Count")
plt.ylabel("Count")
plt.title("TP, TN, FP, and FN vs Report Count")
# plt.xscale("log")  # Log scale for better visualization
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("crosswalk_loop_counts.png")
plt.show()
