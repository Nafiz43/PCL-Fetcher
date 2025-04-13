# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
# This file contains some example run commands
# --reports_to_process=-1 means it will process all the reports in the dataset; 
# provide a valid number to process that many reports

# allowable_models = ["meta.llama3-1-405b-instruct-v1:0", "mistral.mistral-large-2407-v1:0",
#                     "anthropic.claude-3-opus-20240229-v1:0", "anthropic.claude-v2", "meta.llama3-1-70b-instruct-v1:0"]


python3 02_run_llm_bedrock_invoke.py --model_name=meta.llama3-1-405b-instruct-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_invoke.py --model_name=mistral.mistral-large-2407-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_invoke.py --model_name=anthropic.claude-3-opus-20240229-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_invoke.py --model_name=anthropic.claude-v2 --reports_to_process=-1

python3 02_run_llm_bedrock_invoke.py --model_name=meta.llama3-1-70b-instruct-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_invoke.py --model_name=anthropic.claude-3-7-sonnet-20250219-v1:0 --reports_to_process=-1


# FOR COVERSE:

python3 02_run_llm_bedrock_converse.py --model_name=anthropic.claude-3-5-haiku-20241022-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_converse.py --model_name=anthropic.claude-3-7-sonnet-20250219-v1:0 --reports_to_process=-1

python3 02_run_llm_bedrock_converse.py --model_name=anthropic.claude-3-5-sonnet-20241022-v2:0 --reports_to_process=-1



python3 02_run_llm_bedrock_converse_CoT.py --model_name=anthropic.claude-3-5-haiku-20241022-v1:0 --reports_to_process=-1



# python3 run_llm.py --model_name=meditron:70b --reports_to_process=-1

# python3 run_llm.py --model_name=tinyllama --reports_to_process=-1

# python3 run_llm.py --model_name=llama3.3:70b --reports_to_process=-1

# python3 run_llm.py --model_name=qordmlwls/llama3.1-medical:latest --reports_to_process=-1

# python3 run_llm.py --model_name=deepseek-r1:1.5b --reports_to_process=-1

# python3 run_llm.py --model_name=deepseek-r1:7b --reports_to_process=-1

# python3 run_llm.py --model_name=deepseek-r1:70b --reports_to_process=-1

# python3 run_llm.py --model_name=medllama2:latest --reports_to_process=-1

# python3 run_llm.py --model_name=mistral-nemo:latest --reports_to_process=-1

# python3 run_llm.py --model_name=qordmlwls/llama3.1-medical:latest --reports_to_process=-1


