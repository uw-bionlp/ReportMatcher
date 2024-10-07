
import pickle
import json
import os
import pandas as pd
import subprocess
import prompts

use_gpt_prompt = True
base_prompt = False


def format_label(label):
    if label == 0:
        if use_gpt_prompt:
            label_text = 'False'
        else:
            label_text = 'NO'
    elif label == 1:
        if use_gpt_prompt:
            label_text = 'True'
        else:
            label_text = 'YES'
    else:
        print("error: invalid label")
    return f"<answer start> {label_text} <answer end>"


def create_dataset(data, data_savepath, split):
    os.makedirs(data_savepath, exist_ok=True)

    all_records = []
    all_prompts = []
    all_prompt_ids = []

    for row_id, row in data.iterrows():
        
        ind_txt, cand_txt = row['input'].split('[SEP]')

        if use_gpt_prompt:
            if base_prompt:
                input_context = prompts.gpt_base_prompt
            else:
                input_context = prompts.gpt_adv_prompt
            input_context += f'''\nReport A: \n{ind_txt.strip()} \nReport B: \n{cand_txt.strip()}'''

            label = row['label']

            record_structure = [ { "content": input_context, "role": "user" }, { "content": format_label(label), "role": "assistant" } ]
            
            all_records.append(record_structure)
            all_prompts.append(input_context)
            
        else:
            sys_msg = prompts.custom_prompt["system"]
            input_context =  prompts.custom_prompt["user"]
            input_context += f'''\nPatient Radiology report recommendation: {ind_txt.strip()} \nCandidate text: {cand_txt.strip()}'''
            label = row['label']

            record_structure = [ { "content": sys_msg, "role": "system" }, { "content": input_context, "role": "user" }, { "content": format_label(label), "role": "assistant" } ]
            
            all_records.append(record_structure)
            all_prompts.append(sys_msg + input_context)

        all_prompt_ids.append(row['cand_key'])

    split_df = pd.DataFrame(list(zip(all_prompts, all_prompt_ids, all_records)), columns=['prompt', 'prompt_id', 'messages'])
    
    split_df.to_json(f"{data_savepath}/{split}.json", orient='records', lines=True)



def create_test_dataset(data, data_savepath):
    os.makedirs(data_savepath, exist_ok=True)

    cand_ids = []
    all_prompts = []
    all_labels = []

    for _, row in data.iterrows():

        ind_txt, cand_txt = row['input'].split('[SEP]')

        if use_gpt_prompt:
            if base_prompt:
                input_context = prompts.gpt_base_prompt
            else:
                input_context = prompts.gpt_adv_prompt
            input_context += f'''\nReport A: \n{ind_txt.strip()} \nReport B: \n{cand_txt.strip()}'''

        else:
            sys_msg = prompts.custom_prompt["system"] #ignore this value. inference.py will take care of integrating the system message in the correct format for the test data
            input_context = prompts.custom_prompt["user"]
            input_context += f'''\nPatient Radiology report recommendation: {ind_txt.strip()} \nCandidate text: {cand_txt.strip()}'''
        
        all_labels.append(format_label(row['label']))
        
        all_prompts.append(input_context)
        cand_ids.append(row['cand_key'])

    split_df = pd.DataFrame(list(zip(cand_ids, all_prompts, all_labels)), columns=['cand_ids', 'prompt', 'label'])
    split_df.to_csv(data_savepath+ "test.csv", index=False)
    
    

if __name__ == "__main__":

    input_data = "projects/report_matcher/qanda/train-test-data/latest/indfulltext_candfulltext/full_data.csv"
    full_input_data = pd.read_csv(input_data)

    with open("projects/report_matcher/qanda/train-test-data/crossvalfolds.pkl", "rb") as fh:
        crossvalfolds = pickle.load(fh)
    
    results_dir = "data/Meta-Llama-3-8B-Instruct/results/"
    fold_ids = [0,1,2,3,4,0,1,2,3,4]
    for i in range(0,5):
        print(f"\n\n**************************** RUNNING FOLD {str(i)} ********************************", True, results_dir+ f"metrics_log.txt")
        
        train_fold = fold_ids[i:i+3]
        dev_fold = fold_ids[i+3]
        test_fold = fold_ids[i+4]
    
        print(f"\ntrain = {train_fold}, dev = {dev_fold}, test = {test_fold}", True, results_dir+ f"metrics_log.txt")
        train_index_keys = crossvalfolds["split"+str(train_fold[0])] + crossvalfolds["split"+str(train_fold[1])] + crossvalfolds["split"+str(train_fold[2])]    
        val_index_keys = crossvalfolds["split"+str(dev_fold)]   
        test_index_keys = crossvalfolds["split"+str(test_fold)]

        train_index_keys.extend(val_index_keys)
        train_input_data = full_input_data[full_input_data['index_key'].isin(train_index_keys)]
        create_dataset(train_input_data, f"alignment-handbook/custom_dataset/train/fold_{str(i)}/", 'train')

        test_input_data = full_input_data[full_input_data['index_key'].isin(test_index_keys)]
        create_dataset(test_input_data, f"alignment-handbook/custom_dataset/train/fold_{str(i)}/", 'test') # set SFTTrainer.args.do_eval = False. Only to make trl train script run without erroring
        create_test_dataset(test_input_data, f"alignment-handbook/custom_dataset/test/fold_{str(i)}/")