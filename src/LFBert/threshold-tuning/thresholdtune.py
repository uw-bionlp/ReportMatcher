####################################################################
# based on different threshold cutoffs, computes P, R, F1 at patient-level
# IMPORTANT: Before running this script, make sure to fetch all the prediction files into the folder {exp_path}thresholdtune/predictions/
#### These prediction files are the top performing epoch level predictions
#### Rename these files in the format fold<Fold_id>_predictions.csv
####################################################################


import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, f1_score

sys.path.insert(0, "projects/report_matcher/report_matcher_src_code/src/helperfactory/")


def get_scores(select_thresholds=None):
    
    import patientlevel_metrics
    exp_path = "results/latestWOdev60_correct/"

    if not select_thresholds:
        thresholds = [np.round(ti, 3) for ti in np.arange(0.005, 1.0, 0.005).tolist()]
        # thresholds.extend([np.round(ti, 3) for ti in np.arange(0.1, 1.0, 0.005).tolist()])
    else:
        thresholds = select_thresholds

    thresh_columns = []
    for thres in thresholds:
        for met in ["TP", "FP", "TN", "FN", "P", "R", "F1"]:
            thresh_columns.append(f"{str(thres)}_{met}")

    folds = [f"fold_{str(fold_id)}" for fold_id in range(0, 5)]

    thresold_f1s = pd.DataFrame(0.0, index= folds, columns = thresh_columns)

    thresold_f1s['default'] = 0.0

    thresholdwise_scores = {}

    results = pd.DataFrame()
    for fold_id in range(0, 5):
        fold_results = pd.read_csv(f"{exp_path}thresholdtune/predictions/fold{str(fold_id)}_predictions.csv")
        results = pd.concat([results, fold_results])
        
    results['orig_pred'] = results['pred']
    report = classification_report(results['gt'], results['pred'], target_names=['Class 0', 'Class 1'])
    # print(f"---------original report for fold {str(fold_id)} --------")
    # print(report)
    orig_scores, results_msgbuf, patient_results_df = patientlevel_metrics.get_patient_level_metrics(results, "", "", resultsfilename="")
    print("", orig_scores)
    print("--default cutoff--\n ", results_msgbuf)
    thresold_f1s.loc[f"fold_{str(fold_id)}", 'default'] = orig_scores["F1"]
    print("----------------------------------------------")

    threshold_max_f1 = orig_scores["F1"]
    f1_score_per_thresh = {}
    best_t = 0
    for t in thresholds:
        print(f"\n--computing the metrics for threshold = {t}--")
        predictions = (results['prob'].values >= t).astype(int).tolist()
        results['pred'] = predictions

        report = classification_report(results['gt'], results['pred'], target_names=['Class 0', 'Class 1'])
        if select_thresholds:
            results.to_csv(f"{exp_path}thresholdtune/predictions/thresh_{t}_preds.csv")

        scores, results_msgbuf, patient_results_df = patientlevel_metrics.get_patient_level_metrics(results, "", "", resultsfilename="")
        thresholdwise_scores[t] = scores
        for score_metric in scores:
            colname = f"{str(t)}_{score_metric}"
            thresold_f1s.loc[f"fold_{str(fold_id)}", colname] = scores[score_metric]
        print(results_msgbuf)
        f1_score_per_thresh[t] = scores['F1']
        if scores['F1'] > threshold_max_f1:
            threshold_max_f1 = scores['F1']
            best_t = t

    print("-"*80)
    print(f"best F1 = {threshold_max_f1} for threshold = {best_t}")
    print(f1_score_per_thresh)

    if not select_thresholds:
        thresold_f1s.to_csv(f"{exp_path}thresholdtune/thresholdwise_f1s.csv")

    return thresholdwise_scores

        






