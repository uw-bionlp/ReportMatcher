# Report Matcher

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


## In-context learning using GPT-4
- Prompts used in this paper are described below

### Baseline Prompt
```
Question:
You are a board-certified radiologist. You will compare Report A and Report B. 
Your goal is to check whether if Report B is a proper follow-up for Report A 
At the end of your answer, you should include "True" or "False". Your answer should be no longer than 5 sentences.

Answer
```

### Advanced Prompt
```
Question:
You are a board-certified radiologist. You will compare Report A and Report B mostly focusing on information from the sentence in Report A which explicitly suggests a follow-up examination.
Your goal is to check whether Report B is a proper follow-up of Report A.
While a proper follow-up does not always have to use the same imaging test, same day evaluations are not considered as a correct follow-up.

Note: Modality types do not need to match the recommended imaging test, if it can still qualify as a substitute.

After analyzing both reports, return a python list of two elements where you will determine True or False for the following two issues respectively:
1) **reasonable timeframe** (ignore the recommended timeframe and make the decision based on your clinical expertise)
2) provide **updates for the recommendation** from Report A.

At the end of your answer, you should include the following output: [True, False]. Your answer should be no longer than 5 sentences.

Answer: Let’s think step by step.
```

## Significance tests
- The first step is to consolidate the results in a single table
- Run the rm_sigtest.py by passing the consolidated results table
- When running the significance test code, create the bootstrap samples by enabling create_bootstrapsamples in the rm_sigtest.py code 
