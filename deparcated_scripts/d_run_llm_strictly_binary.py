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
from langchain.llms import Ollama
import sys
import random

# data = pd.read_csv('data/Labeled/labels_v2.csv')
data = pd.read_csv('data/labels.csv')

questions = pd.read_csv('data/PCL_Questions_V2.csv')
total_report_count = len(data)

temp = 0
prompt_technique = "base"
prompt_template = "I am going to give you a radiology report. Then I am going to ask you several questions about it. I would like you to answer if a particular type of radiology study or procedure was performed. Please answer with a 1 if the study or procedure was performed. Please answer 0 if the study or procedure was not performed or was not documented in the report. Please answer only with a 1 or 0 without additional words including justification. "

allowable_models = ["meditron:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M"]

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
            writer.writerow(["timestamp", "question", "answer", "model_name"])

    cnt = 0
    # print(questions)

    for index, row in data.iterrows():
        actual_annotation = ""
        for i in range(0, 39):
            
            query = prompt_template + 'REPORT: ' + row['Report_Text'].replace('\n', ' ### ').lower().split('attestation')[0] + '     TASK:' + str(questions.iloc[i]['Questions']) + '. AGAIN, answer either "1" or "0"'
            # print(query)
            ollama = Ollama(model=model_name, temperature=temp, top_k=10, top_p=10)
            response = random.randint(0, 1)  # Random response for testing; comment it when running actual models
            # response = ollama(query) # Uncomment it when doing testing
            
            answer = response

            # print("\n\n> Question:")
            # print(query)
            # print("\n> Answer:")
            # print(answer)

            with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, query, answer, model_name])

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
