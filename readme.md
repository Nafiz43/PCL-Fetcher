# Procedural Case Log Project

This README provides instructions for setting up and running the Procedural Case Log project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Running the Project](#running-the-project)
4. [Extracting Additional Information](#extracting-additional-information)
5. [Generating Comparative Results](#generating-comparative-results)

## System Requirements
The system needs a GPU to get a faster response from the models. The amount of VRAM required depends on the model that we want to run. Here is an estimate:
7B model requires ~4 GB
13B model requires ~8 GB
30B model needs ~16 GB
65B model needs ~32 GB

## Installation

1. Install all required packages and dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install Ollama:
   - Visit [https://ollama.com/](https://ollama.com/) and follow the installation instructions.

3. Download the Ollama models you intend to use locally.

## Running the Project

1. Place your annotated dataset inside the `data` directory.

2. Rename the dataset as `PCL_p.csv` to maintain consistency

3. Run the following command to derive responses from LLMs:
   ```
   python3 run_llm.py --model_name=mixtral:8x7b-instruct-v0.1-q4_K_M --model_type=mistral --save_qa
   ```

   Command breakdown:
   - `--model_name`: Name of the model to run (e.g., mixtral:8x7b-instruct-v0.1-q4_K_M)
   - `--reports_to_process=-1`: Number of reports to process. It accepts a valid integer. By default=-1; it will process all the reports. 

   LLM responses will be stored in the `local_chat_history` directory.

   To get voted response from the LLMs:
   Run the following command to derive responses based on majority voting:
   ```
   python3 run_llm.py --model_1=llama3.2:latest --model_2=mistral --model_3=llama3.1:latest --reports_to_process=-1
   ```

   Command breakdown:
   - `--model_1`: Name of the FIRST model to run (e.g., mixtral:8x7b-instruct-v0.1-q4_K_M)
   - `--model_2`: Name of the SECOND to run (e.g., mistral)
   - `--model_3`: Name of the THIRD to run (e.g., llama3.1:latest)
   - `--reports_to_process=-1`: Number of reports to process. It accepts a valid integer. By default=-1; it will process all the reports. 

   LLM responses will be stored in the `local_chat_history` directory.
   


## Calculating the evaluation metrics
1. Open the run_evaluation_weighted.py file if you want weighted average or open run_evaluation_macro.py if you want macro averaging.

2. Update the name of the following variables:

   `file_containing_ground_truth`= It should contain the path to the ground truth data (i,g: 'data/PCL_p.csv')

3. Run the following command in the shell:
   ```
   python3 run_evaluation_weighted.py --reports_to_process=-1
   ```
   OR,
   ```
   python3 run_evaluation_macro.py --reports_to_process=-1
   ```
   Command breakdown:
   - `--reports_to_process=-1`: Number of reports to process. It accepts a valid integer. By default=-1; it will process all the reports. 


4. Results will be stored in `result.csv` file


However, if you want to see block-by-block output from the code, do the following:
1. Open the `evaluation.ipynb` notebook

2. Activate the localGPT enviroment 

3. In the third cell of the notebook, rename the LLM response file

4. Run all the cells sequenrially.

5. Result will be stored in `result.csv` file


## Docker Commands:
1. Building the Docker Image:
   ```
      sudo docker build -t pcl-container .
   ```
   This command builds a docker image in the name of "pcl-container"


2. To get inside of the docker image:
   ```
   sudo docker run -it --rm --gpus=all pcl-container /bin/bash
   ```
      This command access the docker image while enabling the GPU access

3. We have to run the ollama server:
   ```
   ollama serve &
   ```

   ollama server is up; lets now install the LLM model, with which we want to run the experiment:

   I. Let's install llama3.2:latest (change the model as per the need):
   ```
   ollama run llama3.2:latest
   ```
   As ollama server is up and the model is installed, now we can run the commands as we have run earlier (non-dockerized version); we dont need to install anything :)


4. To copy the `results` folder from the docker image:
   
   I. Open another terminal

   II. Check the name of the docker image:
   ```
   sudo docker ps
   ```

   III. Hit the following command:
   ```
   docker cp DOCKER_IMAGE:/app/results /home/nikhan/Data/Case_Log_Data/Procedural-Case-Log
   ```

      here, replace 'DOCKER_IMAGE' by the actual name of the docker image


5. To get out of the docker:
   ```
   exit
   ```

For AWS Server:
   Use the `checkenv` conda enviroment

For Bedrock Enviroment:
1. Run the following to store the aws credentials:
   `
   mkdir -p ~/.aws && cat <<EOL > ~/.aws/credentials
   [default]
   aws_access_key_id=XXXXXXXXXXXXXXX
   aws_secret_access_key=yyyyyyyyyy
   aws_session_token=xxxxxxx
   EOL
   `

2. To edit the command manually:
   `code ~/.aws/credentials` (this will open the credentials file in VSCODE)
3. For running conversational LLMs using the bedrock enviroment, run the following command:
   `python3 run_llm_bedrock_converse.py`
4. For running non-conversational LLMs using the bedrock enviroment:
   `python3 run_llm_bedrock_invoke.py`


## For running Ollama from pre-built binary:
1. To start the server in background:
```
    OLLAMA_MODELS=/mnt/data/nafiz43 ./ollama serve&
```
2. run a local model afterwards
```
    ./ollama run llama2
```
3. To kill the ollama server:
```
   pkill ollama
```