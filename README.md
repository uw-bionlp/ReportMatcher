# Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches (Accepted at LREC 2026)

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

---

## Prompt Configurations

This project evaluates four prompt configurations for radiology follow-up identification.

---

### 1️⃣ Base Prompt for both GPT-4o and GPT-OSS-20B

```text
You are an expert AI assistant designed to support radiologists in their clinical decision-making processes.

Your primary task is to compare two radiology reports, "Report A" and "Report B," to determine if Report B represents a proper follow-up to Report A.

Your output must be a JSON object with the following structure:
{
  "is_followup": 1,
  "reasoning": "EXPLANATION"
}

- "is_followup": Will be 1 if Report B is a proper follow-up to Report A, and 0 if it is not.
- "reasoning": Must provide a concise, evidence-based explanation for your decision.

This explanation must include direct excerpts from both Report A and Report B that specifically informed your conclusion.

Highlight the key findings, recommendations, or concerns from Report A that are addressed, monitored, or further investigated in Report B.
Conversely, if it's not a follow-up, explain why by pointing to the lack of connection or conflicting information.
Explanation should not exceed 5 sentences.
```

---

### 2️⃣ GPT-OSS-20B Additional Instruction (only added to GPT-OSS-20B prompt)

```text
Note that if any of the findings related to recommendation is addressed, it qualifies as a proper follow-up report.

Report B may still qualify as a follow-up of Report A even though it addresses a subset of findings related to the recommendation.
```

This instruction is appended to the base prompt when evaluating GPT-OSS-20B.

---

### 3️⃣ Base Setting (Full Report Input)

```text
Input:
- Full Index Report (with metadata)
- Full Candidate Report (with metadata)

Prompt:
Use the GPT-4o Base Prompt (above).

Output:
{
  "is_followup": 0 or 1,
  "reasoning": "Concise evidence-based explanation with direct excerpts"
}
```
---

### 4️⃣ Advanced Setting (Recommendation-Aware Prompting)

```text
Input:
- Condensed Index Report:
    - Metadata
    - Recommendation sentence
    - The sentence immediately preceding the recommendation
- Full Candidate Report (with metadata)

Prompt:
Use the GPT-4o Base Prompt (above).
Optionally append GPT-OSS-20B Additional Instruction.

Output:
{
  "is_followup": 0 or 1,
  "reasoning": "Concise evidence-based explanation with direct excerpts"
}
```

**Key Difference:**
The model is explicitly exposed to the actionable recommendation context rather than the entire Index Report.

**Effect:**
- Reduces anchoring on unrelated findings  
- Increases attention to recommendation fulfillment  
- Improves discrimination between true follow-up and incidental overlap  

---


## Significance tests
- The first step is to consolidate the results in a single table
- Run the rm_sigtest.py by passing the consolidated results table
- When running the significance test code, create the bootstrap samples by enabling create_bootstrapsamples in the rm_sigtest.py code 
