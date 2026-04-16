# Procedural Case Log (PCL) Fetcher

PCL Fetcher is a research workflow for evaluating how well large language models (LLMs) can classify radiology/procedural reports into a predefined set of procedure categories.

At a high level, the pipeline:
1. Reads a ground-truth dataset of annotated reports.
2. Prompts an LLM with each report + procedural question list.
3. Saves model outputs and reasoning.
4. Computes evaluation metrics (precision/recall/F1, confusion counts, and aggregates).
5. Produces analysis tables and visualizations.

---

## Table of Contents

- [1) Project Goals](#1-project-goals)
- [2) Repository Layout](#2-repository-layout)
- [3) Requirements](#3-requirements)
- [4) Quick Start (Local / Ollama)](#4-quick-start-local--ollama)
- [5) Input Data Expectations](#5-input-data-expectations)
- [6) Running Inference](#6-running-inference)
- [7) Running Evaluation](#7-running-evaluation)
- [8) Analysis and Visualization Scripts](#8-analysis-and-visualization-scripts)
- [9) Using AWS Bedrock](#9-using-aws-bedrock)
- [10) Docker Workflow](#10-docker-workflow)
- [11) Output Artifacts](#11-output-artifacts)
- [12) Troubleshooting](#12-troubleshooting)
- [13) Privacy & Security Notes](#13-privacy--security-notes)
- [14) Contributing](#14-contributing)
- [15) Contact](#15-contact)
- [16) License](#16-license)

---

## 1) Project Goals

This project is designed to benchmark and compare:
- Prompting strategies (Instruction Prompting vs Chain-of-Thought style prompting).
- Model families (local models, cloud-hosted models, crosswalk baselines).
- Per-question and aggregate procedure classification performance.

The current question set lives in:
- `prompt-files/PCL_Questions_V5.csv`
- `prompt-files/PCL_Questions_V5_CoT.csv`

---

## 2) Repository Layout

```text
.
├── code/                         # Main scripts for inference, evaluation, and analysis
├── prompt-files/                 # Question banks and formatted prompt appendix content
├── figs/                         # Generated figure assets used in analysis/paper outputs
├── static/, index.html           # Static web assets (project page / presentation assets)
├── deparcated_scripts/           # Legacy scripts retained for reference
├── Dockerfile                    # Containerized runtime
└── requirements.txt              # Python dependencies
```

---

## 3) Requirements

### Hardware

GPU is recommended for local inference speed. Approximate VRAM guidance:
- 7B model: ~4 GB
- 13B model: ~8 GB
- 30B model: ~16 GB
- 65B model: ~32 GB

### Software

- Python 3.10+
- Conda (recommended, optional)
- `pip`
- [Ollama](https://ollama.com/) (for local model execution)

---

## 4) Quick Start (Local / Ollama)

### 4.1 Create and activate environment

```bash
conda create -n pcl-fetcher python=3.10 -y
conda activate pcl-fetcher
```

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Start Ollama and pull a model

```bash
ollama serve &
ollama run llama3.2:latest
```

Replace `llama3.2:latest` with your preferred model tag.

---

## 5) Input Data Expectations

Create a `data/` directory and place your annotated file there (commonly named `ground-truth.csv`).

Many scripts expect columns such as:
- `Accession Number`
- `Report Text`
- `Resident`
- Procedure label columns (question-aligned binary labels)

> Note: exact required columns vary slightly by script. If a script drops unknown columns with `errors='ignore'`, extra metadata columns are tolerated.

---

## 6) Running Inference

Main local inference script:

```bash
python3 code/01_run_llm.py \
  --model_name="MODEL_NAME" \
  --prompting_method="IP" \
  --reports_to_process=-1
```

### CLI arguments

- `--model_name`: model identifier (for example, an Ollama model tag)
- `--prompting_method`: one of:
  - `IP` (Instruction Prompting)
  - `CoT` (Chain-of-Thought style)
- `--reports_to_process`:
  - `-1` => all reports
  - positive integer => limit to N reports

Generated chat logs are written to:
- `local_chat_history/` (created automatically if missing)

---

## 7) Running Evaluation

The main evaluation logic is in:
- `code/03_run_evaluation.py`
- `code/04_run_evaluation_crosswalk.py`

Typical flow:
1. Ensure your ground-truth CSV path and LLM output path are set correctly in the evaluation script.
2. Run evaluation for all reports:

```bash
python3 code/03_run_evaluation.py --reports_to_process=-1
```

Results are typically written under:
- `results/`

---

## 8) Analysis and Visualization Scripts

Common post-processing scripts include:

- `code/05_performance_visualizer.py`
- `code/05_performance_visualizer_model.py`
- `code/06_collect_question_number.py`
- `code/06_collect_question_numbers_graph.py`
- `code/07_latex_table_gen.py`
- `code/08_time_and_token_calc.py`
- `code/09_cost_calc_usd.py`
- `code/00_report_token_count_distribution.py`
- `code/00_procedure_count_distribution.py`
- `code/00_resident_count_distribution.py`

These scripts generate:
- performance plots
- token/time/cost summaries
- question-level error distributions
- LaTeX tables for manuscripts

---

## 9) Using AWS Bedrock

Bedrock-related runners:
- `code/02_run_llm_bedrock_converse.py`
- `code/02_run_llm_bedrock_invoke.py`

### 9.1 Configure AWS credentials

Use your secure credential workflow (`aws configure`, role-based auth, or env vars).  
If needed for local testing:

```bash
mkdir -p ~/.aws
cat <<EOL > ~/.aws/credentials
[default]
aws_access_key_id=REPLACE_ME
aws_secret_access_key=REPLACE_ME
aws_session_token=REPLACE_ME
EOL
```

### 9.2 Run Bedrock scripts

```bash
python3 code/02_run_llm_bedrock_converse.py --model_name="MODEL_ID" --reports_to_process=-1
python3 code/02_run_llm_bedrock_invoke.py   --model_name="MODEL_ID" --reports_to_process=-1
```

---

## 10) Docker Workflow

### 10.1 Build image

```bash
docker build -t pcl-container .
```

### 10.2 Run container with GPU

```bash
docker run -it --rm --gpus=all pcl-container /bin/bash
```

### 10.3 Run Ollama/model inside container (if configured)

```bash
ollama serve &
ollama run llama3.2:latest
```

### 10.4 Copy results back to host

In another terminal:

```bash
docker ps
docker cp <CONTAINER_ID>:/app/results /path/on/host/results
```

---

## 11) Output Artifacts

Depending on your run path, outputs may include:
- `local_chat_history/*.csv` (LLM question-level outputs)
- `results/*.csv` (evaluation metrics)
- `figs/*.png` (charts/plots)
- LaTeX table text from analysis scripts

---

## 12) Troubleshooting

- **`ModuleNotFoundError`**: re-run `pip install -r requirements.txt`.
- **Ollama connection error**: confirm `ollama serve` is running.
- **Slow inference**: reduce report count (`--reports_to_process=...`) or use a smaller model.
- **Out of memory (GPU/CPU)**: switch to a smaller quantization/model size.
- **CSV column errors**: inspect headers in your input dataset and align with script expectations.

---

## 13) Privacy & Security Notes

- This repository processes radiology report text; your local dataset may contain sensitive information.
- Do **not** commit patient-identifying datasets, raw PHI, or credential files.
- Store cloud credentials securely and rotate keys regularly.
- Prefer de-identified datasets for experimentation whenever possible.

---

## 14) Contributing

Contributions are welcome.

Recommended workflow:
1. Open an issue describing the bug/feature.
2. Create a focused branch.
3. Submit a PR with:
   - problem statement,
   - approach,
   - test/validation evidence,
   - sample outputs (if relevant).

---

## 15) Contact

For questions, contact: Nafiz Imtiaz Khan **nikhan@ucdavis.edu**

---

## 16) License

This project is licensed under the Apache License 2.0. See `LICENSE.txt`.
