# Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches (Acceptd at LREC 2026)

This project evaluates whether a candidate radiology report represents a proper follow-up to a prior index report using large language models.

The task is framed as a structured clinical reasoning problem: given two radiology reports, determine whether the second report meaningfully addresses findings or recommendations from the first report.

---

## Task Definition

**Input**
- Report A, Index Report  
- Report B, Candidate Report  

**Output**

```json
{
  "is_followup": 0 or 1,
  "reasoning": "Concise evidence-based explanation"
}
```

Where:

- `is_followup = 1` → Report B properly follows up on Report A  
- `is_followup = 0` → Report B is not a follow-up  
- `reasoning` must:
  - Be concise (≤ 5 sentences)
  - Include direct excerpts from both reports
  - Explicitly state whether findings or recommendations were addressed  

---

## Clinical Criteria for Follow-up

A candidate report qualifies as a follow-up if it:

- Addresses a recommendation from Report A  
- Monitors previously described findings  
- Performs additional imaging or evaluation suggested in Report A  
- Evaluates a subset of findings tied to a prior recommendation  

A report does **not** qualify if:

- It focuses on unrelated anatomy or clinical concerns  
- It ignores key findings or recommendations from Report A  
- There is no clinical continuity between the reports  

---
 /src: has the source code for the Supervised models

## Supervised models
Each model has a corresponding folder containing source code to train and validate them.

### SVM and Logistic Regression
- contains training and inference pipeline for SVM and LR
- Configuration template embedded in the workbook needs to be modified to be suitable to the model to train and validate

### LLaMa
- should run preparedata.py to create the datasets for training and validation
- finetuning script: train.sh, inference script: inference.py
- Prompts used/tested for the experiments: prompts.py
- Meta-Llama-3-8B-Instruct was used for finetuning. 
- Needs alignment-handbook 0.4.0.dev0 and trl 0.9.6  The changes in the respective folders must be integrated to run the finetuning and inference. 

### Longformer
- LF_crossval_withtestset.py is the script for finetuning and validation
- Threshold_tuning folder has a script and notebook to analyze the best threshold by the fold and visualize the results



## Significance tests
- The first step is to consolidate the results in a single table
- Run the rm_sigtest.py by passing the consolidated results table
- When running the significance test code, create the bootstrap samples by enabling create_bootstrapsamples in the rm_sigtest.py code 
