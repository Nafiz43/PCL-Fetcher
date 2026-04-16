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
# from langchain_ollama import OllamaLLM as Ollama
# import sys
# from pydantic import BaseModel
# import random
import json
import re
import boto3
from _constant_func import *


data = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')
# data = data[83:]

questions = pd.read_csv('prompt-files/PCL_Questions_V5.csv')
total_report_count = len(data)


bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")


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
            writer.writerow(["timestamp", "accession_number", "question", "answer","reason","model_name"])

    cnt = 0
    # print(questions)

    for index, row in data.iterrows():
        print("ACCESSION Number: ", int(row['Accession Number']))

        for i in range(0, 39):
            # ollama.invoke(prompt)
            report_to_pass = row['Report Text'].lower().split('attestation')[0].split('addendum')[0] + '     TASK:'
            query = 'REPORT: ' + report_to_pass+ str(questions.iloc[i]['Questions'])
            print("PRINTING QUERY:", i)
            print(query)
            print("PRINTING QUERY END")

            # ollama = Ollama(model=model_name, temperature=temp)
            # logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
            # res = ollama.invoke(query)

            ################### BEDROCK CONVERSE###################################
            system_prompts = [{"text": ""+prompt_template+""}]
            # model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
            messages = [{
                    "role": "user",
                    "content": [{"text": ""+query+""}]
                }]
            temperature = 0
            # top_k = 250
            # top_p = 0.999
            # max_tokens_to_sample= 2000

            inference_config = {"temperature": temperature}
            # additional_model_fields = {
            #         "top_k": top_k,
            #         "max_tokens": max_tokens_to_sample,
            #         "top_p": top_p
            #         }

                # Send the message.
            res = bedrock_client.converse(
                    modelId=model_name,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config
                    # additionalModelRequestFields=additional_model_fields
                )
            response = res['output']['message']['content'][0]['text']
            # print(res)

            ################### BEDROCK CONVERSE###################################
            json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            # print(json_match.group(0))
            # json_match = None
            if json_match in [None, ""]:
                json_match = {"reason_for_the_label": "NA", "label": 404}
            else:
                json_match = fix_json(json_match.group(0), response)
            if json_match:
                json_text = json.dumps(json_match)  # Convert dictionary to JSON string
                try:
                    classification = ClassificationResponse.model_validate_json(json_text)
                except Exception as e: 
                    print(f"An error occurred: {type(e).__name__}: {e}")
                    print("JSON text:", json_text)
                    classification = ClassificationResponse(reason_for_the_label="NA", label=404)
            else:
                print("No valid JSON found in the res:", response)
                classification = ClassificationResponse(reason_for_the_label="NA", label=404)

            remove_part = 'REPORT: ' + report_to_pass
            query = query.replace(remove_part, "", 1)

            with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, int(row['Accession Number']), query, classification.label, remove_newlines(classification.reason_for_the_label), model_name])
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
