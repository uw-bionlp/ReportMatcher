import transformers
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report
import argparse
import prompts


from itertools import islice

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--fold', required=True)

args = parser.parse_args()


test_path = f"alignment-handbook/custom_dataset/test/{args.fold}/"
fh = open(test_path + 'scores.txt', "a")

fh.write(str(datetime.now()) + "\n")

test_df = pd.read_csv(test_path + "test.csv")

use_gpt_prompt = True
def get_predictions():
    format_test_data  = []
    for _, row in test_df.iterrows():
        msg_formatted = "\{\{ " + row['prompt'] + "\}\}"    
        if use_gpt_prompt:
            format_test_data.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{msg_formatted}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        else:
            sys_msg = "\{\{ " + prompts.custom_prompt["system"] + "\}\}" 

            format_test_data.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{msg_formatted}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


    #load finetuned checkpoints
    model_id = f"alignment-handbook/custom_dataset/checkpoints/{args.fold}/"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    preds = []
    # create batches
    def batched(iterable, n):
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

    # Iterate over test_data in batches of 4
    for batch in tqdm(batched(format_test_data, 4)):
        # Prepare the batch input for the pipeline
        batch_input = [str(message) for message in batch]
        
        # Get outputs for the batch
        outputs = pipeline(
            batch_input,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        for i, output in enumerate(outputs):
            input_message = batch_input[i]
            input_length = len(input_message)
            generated_text = output[0]["generated_text"][input_length:].strip()

            preds.append(generated_text.strip())
            
    test_df['pred'] = preds
    test_df.to_csv(test_path + "preds.csv", index=False)
    return test_df


def get_patient_level_metrics(results, results_dir="", logfilepath="", resultsfilename="", gt_colname = "gt", pred_colname = "pred"):
    results['unique_keys'] = [x.rsplit("_", 1)[0] for x in results["cand_ids"]]
    idx_keys = list(np.unique(results["unique_keys"]))
    len(idx_keys)
    TP=0
    TN=0
    FP=0
    FN=0
    pat_id = []
    tp_np_counter = pd.DataFrame(0, index=idx_keys, columns=['cand_ids', 'gt', 'pred', 'TP', 'FP', 'TN', 'FN'])
    for ikey in idx_keys:
        pat_id.append(ikey)
        cand_uniq_ids = results[results['unique_keys']==ikey]['cand_ids'].values.tolist()
        tp_np_counter.loc[ikey, 'cand_ids'] = ", ".join(cand_uniq_ids) 
        gt = results[results['unique_keys']==ikey][gt_colname].values.tolist()
        tp_np_counter.loc[ikey, 'gt'] = str(gt)
        preds = results[results['unique_keys']==ikey][pred_colname].values.tolist()
        tp_np_counter.loc[ikey, 'pred'] = str(preds)
        
        #compute TP
        cand_gt_idx = gt.index(1) if 1 in gt else -1
        if cand_gt_idx == -1: # all are 0s (no follow up)
            if 1 not in preds:
                TN += 1
                tp_np_counter.loc[ikey, 'TN'] = 1
            else:
                FP += 1
                tp_np_counter.loc[ikey, 'FP'] = 1
        else: # Follow -up exists
            if preds[cand_gt_idx] == 1:
                if 1 in preds[:cand_gt_idx]:
                    FN += 1 #TODO: validate the definition
                    tp_np_counter.loc[ikey, 'FN'] = 1
                else:
                    TP += 1
                    tp_np_counter.loc[ikey, 'TP'] = 1
            else:
                # if (1 not in preds) or (1 in preds[:cand_gt_idx]) or (1 in preds[cand_gt_idx+1:]) :
                FN += 1
                tp_np_counter.loc[ikey, 'FN'] = 1
    
    
    # printmsg(f"\nTP: {str(TP)} FP: {str(FP)} TN: {str(TN)} FN: {str(FN)}", True, logfilepath)
    results_msgbuf = f"\nTP: {str(TP)} FP: {str(FP)} TN: {str(TN)} FN: {str(FN)}"
    if resultsfilename != "":
        tp_np_counter.to_csv(f"{results_dir}/{resultsfilename}_patientresults.csv")
    
    Precision = TP/(TP+FP) if TP+FP > 0 else 0    
    Recall = TP/(TP+FN) if TP+FN > 0 else 0    
    F1 = 2*Precision*Recall/(Precision + Recall) if Precision + Recall > 0 else 0  

    Precision = np.round(Precision, 3)
    Recall = np.round(Recall, 3)
    F1 = np.round(F1, 3)
    scores = {"TP":TP, "FP":FP, "TN":TN, "FN":FN, "P":Precision, "R":Recall, "F1": F1}    
    
    # printmsg(f"\nP: {str(Precision)} R: {str(Recall)} F1: {str(F1)}", True, logfilepath)
    results_msgbuf += f"\nP: {str(Precision)} R: {str(Recall)} F1: {str(F1)}" 
    return scores, results_msgbuf, tp_np_counter

def get_label(text):
    if use_gpt_prompt:
        if text == '<answer start> False <answer end>':
            return 0
        if text == '<answer start> True <answer end>':
            return 1
        else:
            print(f"ERROR: {text}")
    else:
        if text == '<answer start> NO <answer end>':
            return 0
        if text == '<answer start> YES <answer end>':
            return 1
        else:
            print(f"ERROR: {text}")



test_df = get_predictions()

test_df["pred"] = test_df['pred'].apply(get_label)
test_df['label'] = test_df['label'].apply(get_label)

print(classification_report(test_df['label'], test_df['pred']), file=fh)

scores, results_msgbuf, tp_np_counter = get_patient_level_metrics(test_df, results_dir=test_path, logfilepath=test_path+"scores.log", resultsfilename="test", gt_colname = "label", pred_colname = "pred")
print(results_msgbuf, file=fh)
fh.close()