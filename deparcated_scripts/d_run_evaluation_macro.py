#loading the ground truth
import pandas as pd
import os
import sys
import numpy as np

# file_containing_ground_truth = 'data/Labeled/labels_v2.csv'
# file_containing_llm_response = 'local_chat_history/llama3.2:latest_v3.csv'

file_containing_ground_truth = 'data/labels.csv'
file_containing_llm_response = 'local_chat_history/anthropic.claude-v22025-01-30 03_41_10.csv'
cases_save_dir = file_containing_llm_response.replace('local_chat_history/', '')

fp_file_path = 'results/'+cases_save_dir+'_false_positive_cases.txt'
fn_file_path = 'results/'+cases_save_dir+'_false_negative_cases.txt'


data = pd.read_csv(file_containing_ground_truth)
llm_data = pd.read_csv(file_containing_llm_response)

#Deciding how many documnets to process
reports_to_process = -1
# Parse arguments manually
if "--reports_to_process" in sys.argv:
    idx = sys.argv.index("--reports_to_process")
    reports_to_process = int(sys.argv[idx + 1])

print(f"Reports to process: {reports_to_process}")
#Deciding how many documnets to process

total_num_of_questions = 39

### Identifying how many reports to process###
valid_report_count = len(llm_data)//total_num_of_questions
if(reports_to_process==-1):
    reports_to_process = valid_report_count





data = data.head(reports_to_process)

print("Processing Report Count: ", reports_to_process)
#Cleaning the ground truth data

# data = data.drop(columns=['Other'])
# data = data.drop(columns=['Other.1'])
data = data.drop(columns=['Modality'])
data = data.drop(columns=['Exam Code'])
data = data.drop(columns=['Completed'])
data = data.drop(columns=['Exam Description'])
data.fillna(0, inplace=True)

# print(data.head(2))


#loading the LLM response
# llm_data = pd.read_csv(file_containing_llm_response)

# print(len(llm_data))
# xyz = 122
# llm_data = llm_data.head(reports_to_process*total_num_of_questions)


# print(len(data))


print("Column names:", list(data.columns))
k=0
llm_responses = []
llm_reasonings = []
ground_truths = []
actual_questions = []

# if((len(data)-1)!=len(ground_truth)):
#     print("Invalid LLM Data")
#     # break


for index, row in data.iterrows():
    llm_response = []
    ground_truth = []
    actual_question = []

    ground_truth = row.iloc[2:].tolist()
    # print("Ground Truth: ", index, ground_truth)

    for i in range(total_num_of_questions*index, total_num_of_questions*(index+1)):
        llm_response.append(int(llm_data.answer[k]))
        actual_question.append(llm_data.question[k])
        llm_reasonings.append(llm_data.reason[k])
        k=k+1
    # print("LLM Response: ", llm_response)
    ground_truth = [int(num) for num in ground_truth]
    # print("Ground Truth: ", ground_truth)
    # print(len(ground_truth), len(llm_response))
    llm_responses.append(llm_response)
    ground_truths.append(ground_truth)
    actual_questions.append(actual_question)



def calculate_metrics(ground_truth, llm_response):
    TP = TN = FP = FN = 0
    for gt, pred in zip(ground_truth, llm_response):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative

    # Calculate Precision, Recall, and F1 Score
    precision = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    recall = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0
    f1_score = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0


    # Return results as a dictionary
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Macro-Precision': precision,
        'Macro-Recall': recall,
        'Macro-F1-Score': f1_score
    }

def calculate_metrics_with_cases(report_index, ground_truth, llm_response, llm_reason, actual_question):
    TP = TN = FP = FN = 0
    
    # Loop through the ground truth and LLM responses
    for gt, pred, act in zip(ground_truth, llm_response, actual_question):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
            
            # fp_file_path.replace('.csv',"")
            if not os.path.exists(fp_file_path):
                open(fp_file_path, 'w').close()  # Create an empty file

            with open(fp_file_path, 'a') as file: 
                file.write(f"REPORT: {report_index}; QUESTION: {act}; REASON_BEHIND_POSITIVE: {llm_reason}\n")

        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
            # fn_file_path = fn_file_path.replace(".csv", "")
            if not os.path.exists(fn_file_path):
                open(fn_file_path, 'w').close()  # Create an empty file

            with open('results/'+cases_save_dir+'_false_negative_cases.txt', 'a') as file: 
                file.write(f"REPORT: {report_index}; QUESTION: {act}; REASON_BEHIND_NEGATIVE: {llm_reason}\n")

    # Calculate Precision, Recall, and F1 Score
    precision = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    recall = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0
    f1_score = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0


    # Return results as a dictionary
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Macro-Precision': precision,
        'Macro-Recall': recall,
        'Macro-F1-Score': f1_score
    }


metrics_df = pd.DataFrame(columns=['Modality', 'Report', 'Macro-Precision', 'Macro-Recall', 'Macro-F1-Score'])

for report_index in range(len(llm_responses)):
    print(f"Report {report_index}:")
    
    ###########Considering ALL Modalities#####
    llm_response = llm_responses[report_index]
    ground_truth = ground_truths[report_index]
    actual_question = actual_questions[report_index]
    llm_reasoning = llm_reasonings[report_index]
    metrics = calculate_metrics_with_cases(report_index+1, ground_truth, llm_response, llm_reasoning, actual_question)
    # print(metrics)
    
    new_row = pd.DataFrame({
        'Modality': ["All"],  # Wrap the string in a list
        'Report': [report_index], 
        'TP': [metrics['TP']], 
        'TN': [metrics['TN']], 
        'FP': [metrics['FP']], 
        'FN': [metrics['FN']], 
        'Macro-Precision': [metrics['Macro-Precision']],  # Wrap in a list
        'Macro-Recall': [metrics['Macro-Recall']],  # Wrap in a list
        'Macro-F1-Score': [metrics['Macro-F1-Score']]  # Wrap in a list
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    ###########Considering ALL Modalities#####




    ######Considering VascularDiagnosis ############
    llm_response = llm_responses[report_index][0:8]
    ground_truth = ground_truths[report_index][0:8]
    metrics = calculate_metrics(ground_truth, llm_response)
    # print(metrics)
    
    new_row = pd.DataFrame({
        'Modality': ["VascularDiagnosis"],  # Wrap the string in a list
        'Report': [report_index], 
        'TP': [metrics['TP']], 
        'TN': [metrics['TN']], 
        'FP': [metrics['FP']], 
        'FN': [metrics['FN']], 
        'Macro-Precision': [metrics['Macro-Precision']],  # Wrap in a list
        'Macro-Recall': [metrics['Macro-Recall']],  # Wrap in a list
        'Macro-F1-Score': [metrics['Macro-F1-Score']]  # Wrap in a list
    })

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    ######Considering VascularDiagnosis ############





    ######Considering VascularIntervention ############
    llm_response = llm_responses[report_index][8:8+15]
    ground_truth = ground_truths[report_index][8:8+15]
    metrics = calculate_metrics(ground_truth, llm_response)
    # print(metrics)
    
    new_row = pd.DataFrame({
        'Modality': ["VascularIntervention"],  # Wrap the string in a list
        'Report': [report_index], 
        'TP': [metrics['TP']], 
        'TN': [metrics['TN']], 
        'FP': [metrics['FP']], 
        'FN': [metrics['FN']], 
        'Macro-Precision': [metrics['Macro-Precision']],  # Wrap in a list
        'Macro-Recall': [metrics['Macro-Recall']],  # Wrap in a list
        'Macro-F1-Score': [metrics['Macro-F1-Score']]  # Wrap in a list
    })

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    ######Considering VascularIntervention ############




    ######Considering NonVascularIntervention ############
    llm_response = llm_responses[report_index][8+15:8+15+16]
    ground_truth = ground_truths[report_index][8+15:8+15+16]
    metrics = calculate_metrics(ground_truth, llm_response)
    # print(metrics)
    
    new_row = pd.DataFrame({
        'Modality': ["NonVascularIntervention"],  # Wrap the string in a list
        'Report': [report_index], 
        'TP': [metrics['TP']], 
        'TN': [metrics['TN']], 
        'FP': [metrics['FP']], 
        'FN': [metrics['FN']], 
        'Macro-Precision': [metrics['Macro-Precision']],  # Wrap in a list
        'Macro-Recall': [metrics['Macro-Recall']],  # Wrap in a list
        'Macro-F1-Score': [metrics['Macro-F1-Score']]  # Wrap in a list
    })

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    ######Considering NonVascularIntervention ############

# Aggregated metrics calculation across reports
aggregated_metrics = pd.DataFrame(columns=['Modality', 'TP', 'TN', 'FP', 'FN', 'Macro-Precision', 'Macro-Recall', 'Macro-F1-Score'])

# Group by 'Modality' to calculate aggregated metrics
for modality in metrics_df['Modality'].unique():
    print(modality)
    modality_data = metrics_df[metrics_df['Modality'] == modality]
    
    # Calculate aggregated TP, FP, FN, TN
    total_TP = modality_data['TP'].sum()
    total_TN = modality_data['TN'].sum()
    total_FP = modality_data['FP'].sum()
    total_FN = modality_data['FN'].sum()
    
    # Calculate aggregated Precision, Recall, F1-Score
    precision = round(total_TP / (total_TP + total_FP), 2) if (total_TP + total_FP) > 0 else 0
    recall = round(total_TP / (total_TP + total_FN), 2) if (total_TP + total_FN) > 0 else 0
    f1_score = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0



    new_row = pd.DataFrame({
            'Modality': [modality],
            'Report': 'ALL', 
            'TP': [total_TP],
            'TN': [total_TN],
            'FP': [total_FP],
            'FN': [total_FN],
            'Macro-Precision': [precision],
            'Macro-Recall': [recall],
            'Macro-F1-Score': [f1_score]
    })


    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)


# print(metrics_df)
save_file_path = file_containing_llm_response.replace('local_chat_history/', '')

result_directory = 'results/'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

metrics_df.to_csv(result_directory+save_file_path)
print("Result saved in: "+ result_directory+save_file_path)

