import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
from pydantic import BaseModel
import json
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

prompt_template = """I will provide you with a radiology report, followed by several questions about it. Your task is to determine whether a specific radiology study or procedure was performed. Please follow these strict formatting guidelines for your response:

Output must be in valid JSON format with the following keys:

{
  "reason_for_the_label": "A string explaining the reasoning behind the classification.",
  "label": 1 or 0
}

Labeling criteria:
Return 1 if the radiology study or procedure was explicitly mentioned as performed.
Return 0 if the study or procedure was not performed, not documented, or uncertain in the report.
Do not include any additional text or explanations outside the JSON response.
Ensure strict adherence to this format for every response
"""


prompt_template_CoT_start = "I will provide you with a radiology report, followed by a question about whether a specific radiology study or procedure was performed."

prompt_template_CoT_end = """
### **Strict Output Format**
Your response **must** be a **valid JSON object** with the following keys:
```json
{
  "reason_for_the_label": "A concise explanation justifying the classification.",
  "label": 1 or 0
}
"""
def clean_radiology_report(text):
    # Convert text to lowercase and remove unwanted sections
    text = text.lower().split('attestation')[0].split('ammendum')[0].split('final report electronically signed by')[0].split('preliminary report electronically signed by')[0]

    # Define patterns to remove specified fields
    patterns = [
        r'exam date:.*?(?:\n|$)',
        r'comparison:.*?(?:\n|$)',
        r'date of service:.*?(?:\n|$)',
        r'procedural personnel.*?(?:\n|$)',
        r'attending physician\(s\):.*?(?:\n|$)',
        r'fellow physician\(s\):.*?(?:\n|$)',
        r'resident physician\(s\):.*?(?:\n|$)',
        r'advanced practice provider\(s\):.*?(?:\n|$)'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return text


def process_question_for_IP_prompting(text, question, question_number):
    text = text.strip() + '\n TASK:'
    query = 'REPORT: ' + text+ prompt_template + question
    # print("PRINTING QUERY Number:", question_number+1)
    # print(query)
    return query


def process_question_for_CoT_prompting(text, question, question_number):
    text = text.strip() + '\n TASK:'
    query = 'REPORT: ' + text+ prompt_template_CoT_start + question + prompt_template_CoT_end
    # print("PRINTING QUERY Number:", question_number+1)
    # print(query)
    return query


allowable_models = ["meta.llama3-1-405b-instruct-v1:0", "anthropic.claude-3-5-haiku-20241022-v1:0", 
                    "mistral.mistral-large-2407-v1:0", "anthropic.claude-3-opus-20240229-v1:0", "llama3:8b",
                      "anthropic.claude-v2", "meta.llama3-1-70b-instruct-v1:0", "deepseek-r1:1.5b",
                      "llama3.2:3b-instruct-q4_K_M", "mixtral:8x7b-instruct-v0.1-q4_K_M", "qordmlwls/llama3.1-medical:latest",
                      "medllama2:latest", "meditron:70b", "llama3.2:latest", "anthropic.claude-3-7-sonnet-20250219-v1:0", 
                      "anthropic.claude-3-5-sonnet-20241022-v2:0", "tinyllama", "llama3.3:70b", "thewindmom/llama3-med42-70b",
                        "deepseek-r1:7b", "thewindmom/llama3-med42-8b:latest", "mixtral:latest"]

allowable_prompting_methods = ["IP", "CoT"]

# bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

def remove_newlines(text):
    return text.replace("\n", "").replace("\r", "")

# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

def extract_first_binary(var_name):
    match = re.search(r'[01]', var_name)  # Find the first occurrence of '0' or '1'
    return match.group(0) if match else 0


def fix_json(json_input, response):
    """
    Ensures the input is a JSON string or a dictionary and always returns a dictionary.
    If input is a dictionary, return it as-is.
    If input is a valid JSON string, return parsed JSON as a dictionary.
    If input is an invalid JSON string, attempts to fix it by trimming trailing characters.
    """
    # If input is already a dictionary, return it directly
    if isinstance(json_input, dict):
        return json_input  

    # Ensure input is a string (or bytes), otherwise return error JSON
    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"reason_for_the_label": response, "label": extract_first_binary(response)}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"reason_for_the_label": response, "label": extract_first_binary(response)}
    except json.JSONDecodeError:
        pass  # If invalid, proceed with fixing

    # Try trimming trailing characters
    for i in range(len(json_input), 0, -1):  
        try:
            parsed_json = json.loads(json_input[:i])  # Try parsing progressively shorter substrings
            if isinstance(parsed_json, dict):
                return parsed_json  # Ensure it's a dictionary
        except json.JSONDecodeError:
            continue  # Keep trimming

    # If all attempts fail, return error JSON
    return {"reason_for_the_label": response, "label": extract_first_binary(response)}


def get_dataframes(metrics, modality):
    return pd.DataFrame({
            'Modality': [modality],  
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Sensitivity': [metrics['Sensitivity']], 
            'Specificity': [metrics['Specificity']],  
            'Precision': [metrics['Precision']],  
            'F1-Score': [metrics['F1-Score']],
            'FPR': [metrics['FPR']],
            'TPR': [metrics['TPR']],
            'Sensitivity-Weighted': [metrics['Sensitivity-Weighted']],
            'Specificity-Weighted': [metrics['Specificity-Weighted']],
            'Precision-Weighted': [metrics['Precision-Weighted']],
            'F1-Score-Weighted': [metrics['F1-Score-Weighted']],
            
        })
def get_tp_tn_fp_fn(original_labels, given_labels):  
    TP = TN = FP = FN = 0
    for gt, pred in zip(original_labels, given_labels):
        if pred not in {0,1}:
            pred = 0
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
    return TP, TN, FP, FN

######### CODE FOR EVALUATION ####################
averaging_technique = 'weighted' # It is either macro or weighted

def process_metrics_averaging(original_labels, given_labels): #    
    indexes_to_remove = {i for i, value in enumerate(given_labels) if value == 404}
    original_labels = [value for i, value in enumerate(original_labels) if i not in indexes_to_remove]
    given_labels = [value for i, value in enumerate(given_labels) if i not in indexes_to_remove]


    precision = round((precision_score(original_labels, given_labels, average=averaging_technique, zero_division=0))*100, 2)
    sensitivity = round((recall_score(original_labels, given_labels, average=averaging_technique,zero_division=0))*100, 2) #equivalent to recall
    f1 = round((f1_score(original_labels, given_labels, average=averaging_technique,zero_division=0)), 2)

    try:
        tn, fp, fn, tp = get_tp_tn_fp_fn(original_labels, given_labels)
    except:
        tn, fp, fn, tp = 0,0,0,0
    
    specificity_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_1 = tp / (tp + fn) if (tp + fn) > 0 else 0


    # Compute class proportions
    n0 = np.sum(np.array(original_labels) == 0)
    n1 = np.sum(np.array(original_labels) == 1)
    n = n0 + n1

    # Weighted specificity
    specificity = round(((specificity_0 * n0 + specificity_1 * n1) / n)*100, 2)

    return sensitivity, specificity, precision, f1


def process_metrics_non_averaging(TP, TN, FP, FN): #This calculates for only one class (Positive)    
    sensitivity = round((TP / (TP + FN) if (TP + FN) != 0 else 0)*100, 2) #equivalent to recall
    recall = sensitivity

    specificity = round((TN / (TN + FP) if (TN + FP) != 0 else 0)*100, 2)

    precision = round((TP / (TP + FP) if (TP + FP) != 0 else 0)*100, 2)

    f1_score = round(((2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0), 2)
    
    return sensitivity, specificity, precision, f1_score


def calculate_metrics(original_labels, given_labels):
    TP, TN, FP, FN = get_tp_tn_fp_fn(original_labels, given_labels)
    fpr, tpr, thresholds = roc_curve(original_labels, given_labels)

    sensitivity_weighted, specificity_weighted, precision_weighted, f1_weighted = process_metrics_averaging(original_labels, given_labels)    
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


def calculate_metrics_for_crosswalk(ground_truth, llm_response):    
    # Loop through the ground truth and LLM responses for identifying the FP and FN cases
    chunk_size = 39
    for start in range(0, len(ground_truth), chunk_size):
        cnt = 1  # Question counter

        gt_chunk = ground_truth[start:start + chunk_size]
        pred_chunk = llm_response[start:start + chunk_size]

        for gt, pred in zip(gt_chunk, pred_chunk):
            if gt == 0 and pred == 1:
                with open('results/crosswalk_fp.txt', 'a') as file:
                    file.write(f"QUESTION-{cnt}\n")
            elif gt == 1 and pred == 0:
                with open('results/crosswalk_fn.txt', 'a') as file:
                    file.write(f"QUESTION-{cnt}\n")
            cnt += 1  # Increment question counter
    # Loop through the ground truth and LLM responses for identifying the FP and FN cases

    # Loop through the ground truth and LLM responses for calculating the metrics
    TP = TN = FP = FN = 0
    cnt=1
    for gt, pred in zip(ground_truth, llm_response):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
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

######### CODE FOR EVALUATION ####################
