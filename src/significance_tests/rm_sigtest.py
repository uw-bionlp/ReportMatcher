


import pandas as pd
import pickle
import json
import os
from collections import defaultdict
import random
import numpy as np
import tqdm
import sys
sys.path.append("/home/NETID/gramacha/projects/report_matcher/qanda/")
import helperfactory.patientlevel_metrics as pat

from datetime import datetime 

samplescores_dir = f"/home/NETID/gramacha/projects/report_matcher/qanda/sig test/boostrap_sample_scores_{datetime.now()}/"
os.makedirs(samplescores_dir, exist_ok = True)

all_results_table = pd.DataFrame()

def compute_init_f1(results, pred_colname, initial = False):
    scores, pat_scores_msg, pat_scores_table = pat.get_patient_level_metrics(results, gt_colname='label', pred_colname=pred_colname)
    
    print(pat_scores_msg)

    scores['specificity'] = np.round((scores['TN']/(scores['TN'] + scores['FP'])), 3)
    return {"P": scores["P"], "R": scores["R"], "F1": scores["F1"], "specificity": scores['specificity']}, pat_scores_table

def compute_f1(results):
  
  total_TP = results['TP'].sum()
  total_FP = results['FP'].sum()
  total_TN = results['TN'].sum()
  total_FN = results['FN'].sum()
  
  #compute overall performance
  p = total_TP/(total_TP + total_FP)
  r = total_TP/(total_TP + total_FN)
  f1 = np.round((2*p*r)/(p+r), 3)

  sensitivity = total_TP/(total_TP + total_FN)
  specificity = total_TN/(total_TN + total_FP)
  # if initial:
  #   print(f"TP = {total_TP} FP = {total_FP} TN = {total_TN} FN = {total_FN} \nP = {np.round(p, 3)} R = {np.round(r, 3)} F1 = {np.round(f1, 3)}")
  
  return {"P": p, "R": r, "F1": f1, "specificity": specificity}


def create_bootstrapsamples(all_ids, save_file, n = 100, b=10000):
  bootstraps = {}
  all_sampledids = []
  for i_iter in range(b):
    sampled_ids = []
    # for iter_n in range(n):
    #    sampled_ids.append(random.sample(all_ids, 1)[0])
    choice_ids = random.choice(range(n), size = n, replace=True)
    sampled_ids = all_ids[choice_ids]
    bootstraps[i_iter] = sampled_ids
    all_sampledids.extend(sampled_ids)
    # all_sampledids = list(np.unique(all_sampledids))
  
  with open(save_file, "w") as outfile:
    json.dump(bootstraps, outfile)
  
  missed_ids = [x for x in all_ids if x not in all_sampledids]
  print(f"total ids = {len(all_ids)}, is there anything missed {missed_ids}")
  

   

########################################################################
# non-parametric significance test routine 
########################################################################
def sig_test(results, sysA_result_column, sysB_result_column, sig_log_file, n = 100, b= 10000, bootstrap_samples_exist= True, bootstrapped_samples = ""):

    scores_A, pat_scores_table_A = compute_init_f1(results, sysA_result_column, True)
    print(f"*** system A : {scores_A}", file = sig_log_file)

    scores_B, pat_scores_table_B = compute_init_f1(results, sysB_result_column, True)
    print(f"*** system B : {scores_B}", file = sig_log_file)

    delta = {}
    for k, v in scores_A.items():
      delta[k] = np.round(v - scores_B[k], 4)

    print(f"*** initial delta: *** {delta}", file = sig_log_file)
   
    significant = {"P": 0, "R": 0, "F1": 0, "specificity": 0}
    pvalue = {"P": 0.0, "R": 0.0, "F1": 0.0, "specificity": 0.0} #{"F1": 0.0} #

    all_sampledids = []
    if bootstrap_samples_exist:
        with open(bootstrapped_samples, 'r') as f:
          ids_tosample = json.load(f)
    else:
        ids_tosample = list(np.unique(results_df['index_key']))

    
    bst_row = []
    
    for i_iter in tqdm.tqdm(range(b)):
        if bootstrap_samples_exist:
          sampled_ids = ids_tosample[str(i_iter)]          
        else:
          sampled_ids = random.sample(ids_tosample, n)
          all_sampledids.extend(sampled_ids)
          all_sampledids = list(np.unique(all_sampledids))
       
        sample_iter_A = pat_scores_table_A.loc[sampled_ids, :]
        sample_iter_B = pat_scores_table_B.loc[sampled_ids, :]#pat_scores_table_B[pat_scores_table_B['cand_ids'].isin(sampled_ids)]
        scores_A_b = compute_f1(sample_iter_A)
        scores_B_b = compute_f1(sample_iter_B)
        
        bst_row.append({f'{sysA_result_column}_p' :scores_A_b["P"], f'{sysA_result_column}_r':scores_A_b["R"], f'{sysA_result_column}_f1':scores_A_b["F1"],
                        f'{sysA_result_column}_specificity':scores_A_b["specificity"],
                       f'{sysB_result_column}_p' :scores_B_b["P"], f'{sysB_result_column}_r':scores_B_b["R"], f'{sysB_result_column}_f1':scores_B_b["F1"],
                       f'{sysB_result_column}_specificity':scores_B_b["specificity"]})
        
        for k, v in scores_A_b.items():
          delta_b = v - scores_B_b[k] 

          if delta_b > (2 *delta[k]):
              significant[k] += 1   

        if i_iter > 1 and i_iter % 5000 == 1:
            for k, v in pvalue.items():
              pvalue[k] = significant[k]/i_iter
              if delta[k] < 0:
                pvalue[k] = 1-pvalue[k]
              pvalue[k] = np.round(pvalue[k], 4)
              is_significant = "yes" if pvalue[k] <0.05 else "no" 
              print(f"\t {k} - at iteration {i_iter}, p value = {pvalue[k]}, is_significant = {is_significant}", file = sig_log_file)
    
    #create a table to store scores from each bootstrap sample
    bst_table = pd.DataFrame(bst_row)
    for col in bst_table.columns:
      if col not in all_results_table:
        all_results_table[col] = bst_table[col]
    bst_table.to_csv(f"{samplescores_dir}/{sysA_result_column}-vs-{sysB_result_column}.csv")
    all_results_table.to_csv(f"{samplescores_dir}/all_results.csv")
    
    for k, v in pvalue.items():
      pvalue[k] = significant[k]/b
      if delta[k] < 0:
        pvalue[k] = 1-pvalue[k]
      pvalue[k] = np.round(pvalue[k], 4)

    if not bootstrap_samples_exist:
      #print coverage 
      missed_ids = [x for x in ids_tosample if x not in all_sampledids]
      print(f"total ids = {len(ids_tosample)}, total sampled ids {len(all_sampledids)}, is there anything missed {missed_ids}")
    
    for k, v in pvalue.items():
      print(f"-- {k} -- \nnum significant = \t {significant[k]}", file = sig_log_file)
      print(f"p value = \t {pvalue[k]}", file = sig_log_file)
      is_significant = "yes" if pvalue[k] <0.05 else "no" 
      print(f"is siginificant? {is_significant}\n\n", file = sig_log_file)
    # print("######################################################################\n", file = sig_log_file)
    
  
if __name__ == '__main__':
  # main()
  systems = [
             ("gpt4Best_pred", "gpt4Base_pred", "gpt4Best-Vs-gpt4Base"),
             ("gpt4Best_pred", "svm_fold_best", "gpt4Best-Vs-svm_fold_best"),
             ("gpt4Best_pred", "LR_fold_best", "gpt4Best-Vs-LR_fold_best"),
             ("gpt4Base_pred", "svm_fold_best", "gpt4Base-Vs-svm_fold_best"),
             ("gpt4Base_pred", "LR_fold_best", "gpt4Base-Vs-LR_fold_best"),
             ("svm_fold_best", "LR_fold_best", "svm_fold_best-Vs-LR_fold_best"),
             ("gpt4Best_pred", "llama3_customprompt", "gpt4Best-Vs-llama3_custom"),
             ("gpt4Base_pred", "llama3_customprompt", "gpt4Base-Vs-llama3_custom"),
             ("svm_fold_best", "llama3_customprompt", "svm_fold_best-Vs-llama3_custom"),
             ("LR_fold_best", "llama3_customprompt", "LR-Vs-llama3_custom"),
             ("gpt4Best_pred", "LongFormer_pred", "gpt4Best-Vs-LF"),
             ("gpt4Base_pred", "LongFormer_pred", "gpt4Base-Vs-LF"),
             ("svm_fold_best", "LongFormer_pred", "svm_fold_best-Vs-LF"),
             ("LR_fold_best", "LongFormer_pred", "LR-Vs-LF"),             
             ("llama3_customprompt", "LongFormer_pred", "llama3_custom-Vs-LF"),             
             ]
  

  results_dir = "/home/NETID/gramacha/projects/report_matcher/qanda/sig test/"
  BOOTSTRAP_sample_size_n = 263
  
  sig_log_file = open(results_dir+f"significance_nodev_allmetrics_n{BOOTSTRAP_sample_size_n}.log", "w")
  ids = []

  results_file = results_dir+ "full_data_preds.csv"
  results_df = pd.read_csv(results_file)

  ids = list(np.unique(results_df['index_key']))
  bootstrapped_samples_path = f"/home/NETID/gramacha/projects/report_matcher/qanda/sig test/bootstrapsamples_n{BOOTSTRAP_sample_size_n}_replacement.json"

  # create_bootstrapsamples(ids, bootstrapped_samples_path, n = BOOTSTRAP_sample_size_n, b=10000)


  for system_pair in systems:
    print("***********************************************************************", file = sig_log_file)
    print(f"{system_pair[0]}-VS-{system_pair[1]}", file = sig_log_file)
    print("***********************************************************************", file = sig_log_file)   
    
    sig_test(results_df, system_pair[0], system_pair[1], sig_log_file, n=BOOTSTRAP_sample_size_n, b=10000, bootstrap_samples_exist=True, bootstrapped_samples=bootstrapped_samples_path)
  
  sig_log_file.close()