"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
import sys
from pydantic import BaseModel
import random
import json
import re
import boto3


# data = pd.read_csv('data/Labeled/labels_v2.csv')
data = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv') 
# data = data[37:]

questions = pd.read_csv('data/PCL_Questions_V2.csv')
total_report_count = len(data)

temp = 0
prompt_technique = "base"
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
Ensure strict adherence to this format for every response"""

# model_id = "meta.llama3-1-405b-instruct-v1:0"
	
# model_id = "mistral.mistral-large-2407-v1:0"

# model_id = "anthropic.claude-3-opus-20240229-v1:0"
# model_id = "anthropic.claude-v2"
# model_id = "meta.llama3-1-70b-instruct-v1:0"

allowable_models = ["meta.llama3-1-405b-instruct-v1:0", "mistral.mistral-large-2407-v1:0",
                    "anthropic.claude-3-opus-20240229-v1:0", "anthropic.claude-v2", 
                    "anthropic.claude-3-5-haiku-20241022-v1:0", "meta.llama3-1-70b-instruct-v1:0"]

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")


# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

import json

def fix_json(json_input):
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
        return {"reason_for_the_label": "NA", "label": 404}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"reason_for_the_label": "NA", "label": 404}
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
    return {"reason_for_the_label": "NA", "label": 404}



@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)

def main(model_name, reports_to_process):
    print(f"Received model_name: {model_name}")
    print(f"Received value for reports_to_process: {reports_to_process}")

    global data 

    if(reports_to_process > 0):
        data = data.head(reports_to_process)
        print(f"Processing only {reports_to_process} reports")


    # Your existing logic to handle logging
    log_dir, log_file = "local_chat_history", f"{model_name+datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer","reason","model_name"])

    cnt = 0
    # print(questions)

    for index, row in data.iterrows():
        actual_annotation = ""
        for i in range(0, 39):
            # ollama.invoke(prompt)
            report_to_pass = row['Report Text'].replace('\n', ' ### ').lower().split('attestation')[0] + '     TASK:'
            query = prompt_template + 'REPORT: ' + report_to_pass+ str(questions.iloc[i]['Questions'])
            # print(query)

            # ollama = Ollama(model=model_name, temperature=temp)
            # logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
            # response = ollama.invoke(query)

            ####BEDROCK #####
            payload = {
                "prompt": "Human: "+query+"\n\nAssistant:",
                "max_tokens_to_sample": 2000, # Not supported by LLAMA MODELS; this is for anthropic models
                "temperature": 0  

            }

            payload_bytes = json.dumps(payload).encode("utf-8")

            response = bedrock_client.invoke_model(
                modelId=model_name,
                body=payload_bytes
            )

            response_body = json.loads(response["body"].read())
            # print(response_body)
            # print(type(response_body['generation'])) # this should work with llama models
            # print(type(response_body['completion'])) # this should work with claude models

            res = response_body['completion']
            ####BEDROCK #####




            # print(response)
            
            json_match = re.search(r"\{.*\}", res.strip(), re.DOTALL)
            # print(json_match)
            # json_match = None
            if json_match in [None, ""]:
                json_match = {"reason_for_the_label": "NA", "label": 404}
            else:
                json_match = fix_json(json_match.group(0))
                
            if json_match:
                json_text = json.dumps(json_match)  # Convert dictionary to JSON string
                try:
                    classification = ClassificationResponse.model_validate_json(json_text)
                    # print(classification)
                except json.JSONDecodeError as e:
                    print("Error parsing JSON:", e)
            else:
                print("No valid JSON found in the response:", response)

            remove_part = prompt_template + 'REPORT: ' + report_to_pass

            # Remove it from `query`
            query = query.replace(remove_part, "", 1)
            # print("query: ", query)

            with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, query, classification.label, classification.reason_for_the_label, model_name])
                print("Questions Completed", i, end="\r")

        progress_percentage = ((index+1) / len(data)) * 100
        print(f"Processed {index+1}/{len(data)} reports ({progress_percentage:.2f}% complete)", end="\r")
        # sys.stdout.flush()

    # print("\n")
    print("\nTotal Reports Processed", len(data))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
