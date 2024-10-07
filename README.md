# Report Matcher

 /src: has the source code for the Supervised models

## Supervised models
Each model has a corresponding folder containing source code to train and validate them.

### SVM AND Logistic Regression
- contains training and inference pipeline for SVM and LR
- Configuration template embedded in the workbook needs to be modified to be suitable to the model to train and validate

### LLaMa
- should run preparedata.py to create the datasets for training and validation
- finetuning script: train.sh, inference script: inference.py
- Prompts used/tested for the experiments: prompts.py
- Meta-Llama-3-8B-Instruct was used for finetuning. 
- Needs alignment-handbook 0.4.0.dev0 and trl 0.9.6  The changes in the respective folders must be integrated to run the finetuning and inference. 

### LFBert
- LF_crossval_withtestset.py is the script for finetuning and validation
- Threshold_tuning folder has a script and notebook to analyze the best threshold by the fold and visualize the results


## In-context learning using GPT-4


## Significance tests
- The first step is to consolidate the results in a single table
- Run the rm_sigtest.py by passing the consolidated results table
- When running the significance test code, create the bootstrap samples by enabling create_bootstrapsamples in the rm_sigtest.py code 
