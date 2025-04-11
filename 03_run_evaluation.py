import pandas as pd
import os
import sys
import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from _constant_func import *


file_containing_ground_truth = 'data/Labeled_Reports_2025_02_14_V02.csv'
local_history_directory = 'local_chat_history'
os.makedirs(os.path.dirname("results/"), exist_ok=True) #making sure the directory exists


all_models_results = pd.DataFrame()

def calculate_metrics_with_cases(report_index, ground_truth, llm_response, llm_reason, actual_question, accession_number):
    TP = TN = FP = FN = 0
    
    # Loop through the ground truth and LLM responses
    cnt=1
    for gt, pred, act, reason in zip(ground_truth, llm_response, actual_question, llm_reason):
        if pred not in {0,1}:
            # print("report_index", report_index, "Accession Number",accession_number, "Anomaly; Prediction-Value:", pred, "Ground-Truth-Value: ", gt)
            pred = 0
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive  
            with open(fp_file_path, 'a') as file: 
                file.write(f"ACCESSION_NUMBER: {accession_number}; REPORT: {report_index}; QUESTION-{cnt}: {act}; REASON_BEHIND_POSITIVE: {reason}\n")

        elif gt == 1 and pred == 0:
            FN += 1  # False Negative

            with open('results/'+cases_save_dir+'_false_negative_cases.txt', 'a') as file: 
                file.write(f"ACCESSION_NUMBER: {accession_number}; REPORT: {report_index}; QUESTION-{cnt}: {act}; REASON_BEHIND_NEGATIVE: {reason}\n")

        cnt=cnt+1
    
    fpr, tpr, thresholds = roc_curve(ground_truth, llm_response)

    sensitivity_weighted, specificity_weighted, precision_weighted, f1_weighted = process_metrics_averaging(ground_truth, llm_response)    
    sensitivity, specificity, precision, f1 = process_metrics_non_averaging(TP, TN, FP, FN)

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'FPR': fpr,
        'TPR': tpr,
        'Sensitivity-Weighted': sensitivity_weighted,
        'Specificity-Weighted': specificity_weighted,
        'Precision-Weighted': precision_weighted,
        'F1-Score-Weighted': f1_weighted,
    }



csv_files = [f for f in os.listdir(local_history_directory) if f.endswith('.csv')]
for csv_file in csv_files:
    print("Processing", os.path.join(local_history_directory, csv_file))

    file_containing_llm_response = os.path.join(local_history_directory, csv_file)
    cases_save_dir = file_containing_llm_response.replace(local_history_directory+'/', '')

    fp_file_path = 'results/'+cases_save_dir+'_false_positive_cases.txt'
    fn_file_path = 'results/'+cases_save_dir+'_false_negative_cases.txt'
    open(fp_file_path, 'w').close()  # Create an empty file
    open(fn_file_path, 'w').close()  # Create an empty file


    data = pd.read_csv(file_containing_ground_truth, on_bad_lines='skip')
    llm_data = pd.read_csv(file_containing_llm_response, on_bad_lines='skip')

    #Deciding how many documnets to process
    reports_to_process = -1
    # Parse arguments manually
    if "--reports_to_process" in sys.argv:
        idx = sys.argv.index("--reports_to_process")
        reports_to_process = int(sys.argv[idx + 1])

    # print(f"Reports to process: {reports_to_process}")
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
    data = data.drop(columns=['Resident'])
    data = data.drop(columns=['Completed REG'])
    data = data.drop(columns=['Example Case List'])
    
    
    data.fillna(0, inplace=True)

    #loading the LLM response
    llm_data = pd.read_csv(file_containing_llm_response)
    llm_data = llm_data.head(reports_to_process*total_num_of_questions)

    k=0
    llm_responses = []
    llm_reasonings = []
    ground_truths = []
    actual_questions = []

    vascular_diagonsis_ground_truth = []
    vascular_diagonsis_llm = []

    vascular_intervention_ground_truth = []
    vascular_intervention_llm = []

    non_vascular_intervention_ground_truth = []
    non_vascular_intervention_llm = []

    accession_numbers = []


    # The following for LOOP constructs 4 lists
    for index, row in data.iterrows(): #This iterates through all the report (ground truth dataset)
        llm_response = []
        ground_truth = []
        actual_question = []
        llm_reasoning = []
        # print(row)

        ground_truth = row.iloc[2:].tolist()
        # print("Accession Number", int(row.iloc[0]))
        accession_numbers.append(int(row.iloc[0]))
        # print("Ground Truth: ", index, ground_truth)

        for i in range(total_num_of_questions*index, total_num_of_questions*(index+1)):
            llm_response.append(int(llm_data.answer[k]))
            actual_question.append(llm_data.question[k])
            llm_reasoning.append(llm_data.reason[k]) #This compiles list of reasonings for a particular report
            k=k+1
        # print("LLM Response: ", llm_response)
        ground_truth = [int(num) for num in ground_truth]
        # print("Report: ", index+1)
        # print("Ground Truth: ", ground_truth)
        # print("Actual Labels: ", llm_response)

        # print(len(ground_truth), len(llm_response))
        llm_responses.append(llm_response)
        ground_truths.append(ground_truth)
        actual_questions.append(actual_question)
        llm_reasonings.append(llm_reasoning) #This compiles list of reasonings for all the report. It is a list of list


    metrics_df = pd.DataFrame(columns=['Modality', 'TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'FPR', 'TPR'])

    # print("LLM RESPONSES:", llm_responses)
    for report_index in range(0, len(llm_responses)):        
        ###########Considering ALL Modalities#####
        llm_response = llm_responses[report_index]
        ground_truth = ground_truths[report_index]
        actual_question = actual_questions[report_index]
        llm_reasoning = llm_reasonings[report_index]
        accession_number = accession_numbers[report_index]

        # PRINTING INFO FOR SANITY CHECK
        print("Report: ", report_index+1)
        print("Ground Truth: ", ground_truth)
        print("Actual Labels: ", llm_response)
        print("Accession Number: ", accession_number)
        # PRINTING INFO FOR SANITY CHECK

        metrics = calculate_metrics_with_cases(report_index+1, ground_truth, llm_response, llm_reasoning, actual_question, accession_number)
        # print(metrics)

        new_row = get_dataframes(metrics, "All")
        
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ###########Considering ALL Modalities#####

        ######Considering VascularDiagnosis ############
        llm_response = llm_responses[report_index][0:8]
        ground_truth = ground_truths[report_index][0:8]

        # print(vascular_diagonsis_ground_truth)

        vascular_diagonsis_ground_truth.extend(ground_truth)
        vascular_diagonsis_llm.extend(llm_response)

        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = get_dataframes(metrics, "VascularDiagnosis")

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering VascularDiagnosis ############


        ######Considering VascularIntervention ############
        llm_response = llm_responses[report_index][8:8+15]
        ground_truth = ground_truths[report_index][8:8+15]

        vascular_intervention_ground_truth.extend(ground_truth)
        vascular_intervention_llm.extend(llm_response)

        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = get_dataframes(metrics, "VascularIntervention")
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering VascularIntervention ############


        ######Considering NonVascularIntervention ############
        llm_response = llm_responses[report_index][8+15:8+15+16]
        ground_truth = ground_truths[report_index][8+15:8+15+16]
        non_vascular_intervention_ground_truth.extend(ground_truth)
        non_vascular_intervention_llm.extend(llm_response)


        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = get_dataframes(metrics, "NonVascularIntervention")

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering NonVascularIntervention ############

    # Aggregated metrics calculation across reports
    aggregated_metrics = pd.DataFrame(columns=[])

    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    # print("OUTSIDE",llm_responses)
    for modality in metrics_df['Modality'].unique():
        # print(modality)
        modality_data = metrics_df[metrics_df['Modality'] == modality]
        
        # Calculate aggregated TP, FP, FN, TN
        total_TP = modality_data['TP'].sum()
        total_TN = modality_data['TN'].sum()
        total_FP = modality_data['FP'].sum()
        total_FN = modality_data['FN'].sum()

        if(modality == "All"):
            metrics=calculate_metrics(np.concatenate(ground_truths), np.concatenate(llm_responses))
        elif(modality == "VascularDiagnosis"):
            metrics=calculate_metrics(vascular_diagonsis_ground_truth, vascular_diagonsis_llm)
        elif(modality == "VascularIntervention"):
            metrics=calculate_metrics(vascular_intervention_ground_truth, vascular_intervention_llm)
        elif(modality == "NonVascularIntervention"):
            metrics = calculate_metrics(non_vascular_intervention_ground_truth, non_vascular_intervention_llm)

        new_row = get_dataframes(metrics, [modality])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)


    # print(metrics_df)
    save_file_path = file_containing_llm_response.replace(local_history_directory+'/', '')

    result_directory = 'results/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    
    metrics_df = metrics_df.drop('TPR', axis=1)
    metrics_df = metrics_df.drop('FPR', axis=1)
    metrics_df = metrics_df[len(metrics_df)-4: len(metrics_df)]
    metrics_df["Model-Name"] = save_file_path[:20]

    metrics_df.to_csv(result_directory+save_file_path, index=False)

    all_models_results = pd.concat([all_models_results, metrics_df], ignore_index=True)
    print("Result saved in: "+ result_directory+save_file_path)
    # print(metrics_df)

all_models_results.to_csv('results/all_models.csv', index=False)
print("All Models Result saved in: `all_models.csv` file")
