import numpy as np
import pandas as pd


def printmsg(msg, printotconsole=True, logfilepath=""):
    if printotconsole:
        print(msg)
    if logfilepath != "":
        with open(logfilepath, "a") as text_file:
            text_file.write(msg)

# def get_patient_level_metrics(results, results_dir="", logfilepath="", resultsfilename=""):
#     results['unique_keys'] = [x.rsplit("_", 1)[0] for x in results["cand_key"]]
#     idx_keys = list(np.unique(results["unique_keys"]))
#     len(idx_keys)
#     TP=0
#     TN=0
#     FP=0
#     FN=0
#     pat_id = []
#     tp_np_counter = pd.DataFrame(0, index=idx_keys, columns=['cand_ids', 'gt', 'pred', 'TP', 'FP', 'TN', 'FN'])
#     for ikey in idx_keys:
#         pat_id.append(ikey)
#         cand_uniq_ids = results[results['unique_keys']==ikey]['cand_key'].values.tolist()
#         tp_np_counter.loc[ikey, 'cand_ids'] = ", ".join(cand_uniq_ids) 
#         gt = results[results['unique_keys']==ikey]['gt'].values.tolist()
#         tp_np_counter.loc[ikey, 'gt'] = str(gt)
#         preds = results[results['unique_keys']==ikey]['pred'].values.tolist()
#         tp_np_counter.loc[ikey, 'pred'] = str(preds)
        
#         #compute TP
#         cand_gt_idx = gt.index(1) if 1 in gt else -1
#         if cand_gt_idx == -1: # all are 0s (no follow up)
#             if 1 not in preds:
#                 TN += 1
#                 tp_np_counter.loc[ikey, 'TN'] = 1
#             else:
#                 FP += 1
#                 tp_np_counter.loc[ikey, 'FP'] = 1
#         else: # Follow -up exists
#             if preds[cand_gt_idx] == 1:
#                 if 1 in preds[:cand_gt_idx]:
#                     FN += 1 #TODO: validate the definition
#                     tp_np_counter.loc[ikey, 'FN'] = 1
#                 else:
#                     TP += 1
#                     tp_np_counter.loc[ikey, 'TP'] = 1
#             else:
#                 # if (1 not in preds) or (1 in preds[:cand_gt_idx]) or (1 in preds[cand_gt_idx+1:]) :
#                 FN += 1
#                 tp_np_counter.loc[ikey, 'FN'] = 1
    
    
#     printmsg(f"\nTP: {str(TP)} FP: {str(FP)} TN: {str(TN)} FN: {str(FN)}", True, logfilepath)
#     if resultsfilename != "":
#         tp_np_counter.to_csv(f"{results_dir}/{resultsfilename}_patientresults.csv")
    
#     Precision = TP/(TP+FP) if TP+FP > 0 else 0    
#     Recall = TP/(TP+FN) if TP+FN > 0 else 0    
#     F1 = 2*Precision*Recall/(Precision + Recall) if Precision + Recall > 0 else 0    
#     scores = {"P":Precision, "R":Recall, "F1": F1}    
    
#     printmsg(f"\nP: {str(Precision)} R: {str(Recall)} F1: {str(F1)}", True, logfilepath)   
#     return scores



# get patient level metrics. Input is a predictions file that has a binary prediction value for each index candidate pair

def get_patient_level_metrics(results, results_dir="", logfilepath="", resultsfilename="", gt_colname = "gt", pred_colname = "pred"):
    results['unique_keys'] = [x.rsplit("_", 1)[0] for x in results["cand_key"]]
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
        cand_uniq_ids = results[results['unique_keys']==ikey]['cand_key'].values.tolist()
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
