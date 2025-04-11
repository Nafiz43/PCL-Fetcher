import pandas as pd
import multiprocessing as mp
from sklearn.metrics import precision_score, recall_score, f1_score

def process_metrics(original_labels, crosswalk_labels):
    indexes_to_remove = {i for i, value in enumerate(crosswalk_labels) if value == 404}
    original_labels = [value for i, value in enumerate(original_labels) if i not in indexes_to_remove]
    crosswalk_labels = [value for i, value in enumerate(crosswalk_labels) if i not in indexes_to_remove]

    precision = round(precision_score(original_labels, crosswalk_labels, average='weighted', zero_division=0), 3)
    recall = round(recall_score(original_labels, crosswalk_labels, average='weighted', zero_division=0), 3)
    f1 = round(f1_score(original_labels, crosswalk_labels, average='weighted', zero_division=0), 3)
    return precision, recall, f1

def calculate_metrics(args):
    original_labels, crosswalk_labels, reports_to_process = args
    TP = TN = FP = FN = 0

    for gt, pred in zip(original_labels, crosswalk_labels):
        if gt == 1 and pred == 1:
            TP += 1  
        elif gt == 0 and pred == 0:
            TN += 1  
        elif gt == 0 and pred == 1:
            FP += 1  
        elif gt == 1 and pred == 0:
            FN += 1  

    precision, recall, f1 = process_metrics(original_labels, crosswalk_labels)

    return {
        'Modality': "All",
        'report_count': reports_to_process,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Weighted-Precision': precision,
        'Weighted-Recall': recall,
        'Weighted-F`1-Score': f1
    }

def process_reports(reports_to_process):
    print(f"Processing {reports_to_process} reports")
    
    crosswalk = pd.read_csv('data/cross_walk.csv').fillna(0).drop(columns=['Accession Number', 'Modality', 'Completed', 'Exam Description', 'Report Text', 'Resident'])
    original_report = pd.read_csv('data/labels.csv').fillna(0).drop(columns=['Accession Number', 'Modality', 'Completed', 'Exam Description', 'Report_Text', 'Resident'])

    if reports_to_process > 0:
        original_report = original_report.head(reports_to_process)

    all_original_report_labels, all_crosswalk_labels = [], []

    for _, outer_row in original_report.iterrows():
        exam_code = outer_row['Exam Code']
        original_labels = [int(x) for x in outer_row.iloc[2:].tolist()]
        all_original_report_labels.extend(original_labels)

        matched_row = crosswalk[crosswalk['Exam Code'] == exam_code]
        if not matched_row.empty:
            crosswalk_labels = [int(x) for x in matched_row.iloc[0, 2:].tolist()]
            all_crosswalk_labels.extend(crosswalk_labels)

    return calculate_metrics((all_original_report_labels, all_crosswalk_labels, reports_to_process))

if __name__ == "__main__":
    reference_numbers = [10, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000]
    num_cores = mp.cpu_count()  # Get available CPU cores

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_reports, reference_numbers)

    metrics_df = pd.DataFrame(results)
    print(metrics_df)
    metrics_df.to_csv("crosswalk_performance.csv", index=False)
    print("CSV file saved as crosswalk_performance.csv")
