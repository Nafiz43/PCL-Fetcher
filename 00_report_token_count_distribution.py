import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from _constant_func import clean_radiology_report
from _stat_gen import count_tokens
from scipy.stats import gaussian_kde



df = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')

token_counts_raw = pd.array(df['Report Text'].apply(lambda x: count_tokens(x) if pd.notnull(x) else 0))
token_counts_cleaned = pd.array(df['Report Text'].apply(lambda x: clean_radiology_report(x)).apply(lambda x: count_tokens(x) if pd.notnull(x) else 0))
# print("Token Count Raw", token_counts_raw)

# Example data

def histogram_gen(data, type):
    # Custom bins
    bins = np.arange(0, 2000 + 250, 250)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Histogram
    counts, _ = np.histogram(data, bins=bins)

    # KDE on raw data
    kde = gaussian_kde(data)
    x_vals = np.linspace(0, max(data) + 100, 500)
    kde_vals = kde(x_vals)

    plt.figure(figsize=(6, 4))  

    # Plot histogram
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.6, label='Histogram', color='#93aacf', linewidth=1.6)

    # Add count labels on top of each bar
    for count, x in zip(counts, bin_centers):
        plt.text(x, count + 1, str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Overlay KDE curve
    plt.plot(x_vals, kde_vals * len(data) * (bins[1] - bins[0]), label='KDE Curve', linewidth=2)

    plt.xlabel('Token Count Range')
    plt.ylabel('Report Frequency')
    plt.grid(True)
    plt.xticks(bins)

    saving_path = 'figs/token_histogram-' + type
    plt.savefig(saving_path, bbox_inches='tight', dpi=300)
    print("File saved in", saving_path)
    # plt.show()


histogram_gen(token_counts_raw, "raw")
histogram_gen(token_counts_cleaned, "cleaned")
