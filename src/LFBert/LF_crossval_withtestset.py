import os
#delete this later
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import pandas as pd
import pickle

import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoTokenizer,AutoModel,AutoConfig, BertForSequenceClassification

from transformers import Trainer, TrainingArguments, LongformerTokenizer, LongformerForSequenceClassification
from sklearn.metrics import classification_report,confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from datetime import datetime
import os
from pathlib import Path

import sys
sys.path.append("projects/report_matcher/report_matcher_src_code/src/")
import helperfactory.patientlevel_metrics as pat

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

'''
custom function to print the output and the write to the log file at the simultaneously
'''
def printmsg(msg, printotconsole=True, logfilepath=""):
    if printotconsole:
        print(msg)
    if logfilepath != "":
        with open(logfilepath, "a") as text_file:
            text_file.write(msg)


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


class BertBinaryClassifier(nn.Module):
    def __init__(self, checkpoint, yt_weight, freeze_bert=False):
        super(BertBinaryClassifier, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(checkpoint).cuda()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(last_hidden_state_cls))

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(yt_weight[1])
            loss = loss_fn(logits.view(-1), labels.float())
            return loss, logits
        else:
            return logits

"""
Two routines-
Train-validation pipeline
loop for hyp tuning in main func

def training_pipe(model, train_dataset, valid_dataset, valid_df, checkpoints_save_dir, logfilepath, epochs, lr, train_batchsize, val_batchsize, logs_dir, results_dir, gradient_acc_steps, run_name):#if train_flag:
    training_args = TrainingArguments(
        output_dir=checkpoints_save_dir,
        num_train_epochs=epochs,
        learning_rate = lr,
        per_device_train_batch_size=train_batchsize,
        per_device_eval_batch_size=val_batchsize,
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none",
        logging_dir=logs_dir,
        gradient_accumulation_steps=gradient_acc_steps,
        dataloader_drop_last = False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        probs = torch.sigmoid(torch.tensor(pred.predictions))
        preds = torch.round(probs)
        f1 = f1_score(labels, preds)
        return {
            'f1': f1
        }

    if valid_dataset:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )    
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics
        )

    class GenerateTextCallback(TrainerCallback):
        def __init__(self, trainer, test_data, test_df):
            self.trainer = trainer
            self.test_data = test_data
            self.test_df = test_df
            self.best_f1 = 0
            self.best_epoch = 0

        def getbestepoch(self):
            return self.best_epoch
        def getbestmetric(self):
            return self.best_f1
        
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            predictions = self.trainer.predict(self.test_data)
            # 'predictions' is a tuple, and the logits are stored in the first element of the tuple
            logits = predictions.predictions
            probs = torch.sigmoid(torch.tensor(logits))
            binary_preds = torch.round(probs)
            all_preds = binary_preds.flatten().numpy().tolist()
            
            true_labels = predictions.label_ids
            all_gts = true_labels.flatten()           
        
            gts = [int(x) for x in all_gts]
            preds = [int(x) for x in all_preds]
            rounds =  [x for x in self.test_df['round'].values]
            index_keys = [x for x in self.test_df['index_key'].values]
            cand_keys = [x for x in self.test_df['cand_key'].values]  

            pred_df = pd.DataFrame(list(zip(rounds, index_keys, cand_keys, gts, preds)), columns = ['round','index_key','cand_key','gt','pred'])

            os.makedirs(results_dir + f"eval-epoch_wise/{run_name}/", exist_ok=True)
            pred_df.to_csv(f"{results_dir}/eval-epoch_wise/{run_name}/ep{str(np.round(state.epoch))}_predictions.csv")

            report = classification_report(gts, preds, target_names=['Class 0', 'Class 1'])
            printmsg(f"\nevaluation results for epoch {str(np.round(state.epoch))} \n {report}", True, f"{logs_dir}/{run_name}.txt")
                        
            scores = get_patient_level_metrics(pred_df, results_dir+f"eval-epoch_wise/{run_name}/", f"{logs_dir}/{run_name}.txt", run_name+str(np.round(state.epoch)))
            if scores['F1'] >= self.best_f1:
                self.best_f1 = scores['F1']
                self.best_epoch = int(np.round(state.epoch))


    generate_callback = GenerateTextCallback(trainer, valid_dataset, valid_df)
    trainer.add_callback(generate_callback)
    
    printmsg(f"\n{datetime.now()} started training..", True, logfilepath)
    trainer.train()
    printmsg(f"\n{datetime.now()} completed training.", True, logfilepath)
    bestepoch = generate_callback.getbestepoch()
    bestf1 = generate_callback.getbestmetric()
    return trainer, bestepoch, bestf1
"""
"""
Trains a model for x epochs and saves the checkpoint.
does not require any dev set
"""

def final_training_pipe(model, train_dataset, test_dataset, test_df, valid_dataset, checkpoints_save_dir, logfilepath, epochs, lr, train_batchsize, logs_dir, results_dir, gradient_acc_steps, run_name):#if train_flag:
    training_args = TrainingArguments(
        output_dir=checkpoints_save_dir,
        num_train_epochs=epochs,
        learning_rate = lr,
        per_device_train_batch_size=train_batchsize,
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none",
        logging_dir=logs_dir,
        gradient_accumulation_steps=gradient_acc_steps,
        dataloader_drop_last = False,
        evaluation_strategy="no",
        save_strategy='epoch',
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    class GenerateTextCallback(TrainerCallback):
        def __init__(self, trainer, test_data, test_df, valid_data):
            self.trainer = trainer
            self.test_data = test_data
            self.valid_data = valid_data
            self.test_df = test_df
            self.best_f1 = 0
            self.best_epoch = 0

        def getbestepoch(self):
            return self.best_epoch
        def getbestmetric(self):
            return self.best_f1
        
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

            ## validation set
            if self.valid_data:
                predictions = self.trainer.predict(self.valid_data)
                logits = predictions.predictions
                probs = torch.sigmoid(torch.tensor(logits))
                binary_preds = torch.round(probs)
                all_preds = binary_preds.flatten().numpy().tolist()
                
                true_labels = predictions.label_ids
                all_gts = true_labels.flatten()           
            
                gts = [int(x) for x in all_gts]
                preds = [int(x) for x in all_preds]

                report = classification_report(gts, preds, target_names=['Class 0', 'Class 1'])
                printmsg(f"\n*VAL*evaluation results for epoch {str(state.epoch)} \n {report}", True, f"{logs_dir}/{run_name}.txt")

            ## test set
            predictions = self.trainer.predict(self.test_data)
            logits = predictions.predictions
            probs = torch.sigmoid(torch.tensor(logits))
            binary_preds = torch.round(probs)
            all_preds = binary_preds.flatten().numpy().tolist()

            logits_list = logits.flatten().tolist()
            probs_list = probs.flatten().numpy().tolist()
            
            true_labels = predictions.label_ids
            all_gts = true_labels.flatten()           
        
            gts = [int(x) for x in all_gts]
            preds = [int(x) for x in all_preds]
            rounds =  [x for x in self.test_df['round'].values]
            index_keys = [x for x in self.test_df['index_key'].values]
            cand_keys = [x for x in self.test_df['cand_key'].values]  

            os.makedirs(results_dir + "test-epoch_wise", exist_ok=True)
            os.makedirs(results_dir + f"test-epoch_wise/{run_name}/", exist_ok=True)
            
            pred_df = pd.DataFrame(list(zip(rounds, index_keys, cand_keys, gts, preds, probs_list, logits_list)), columns = ['round','index_key','cand_key','gt','pred', 'prob', 'logit'])
            pred_df.to_csv(f"{results_dir}/test-epoch_wise/{run_name}/ep{str(np.round(state.epoch))}_predictions.csv")
    
            report = classification_report(gts, preds, target_names=['Class 0', 'Class 1'])
            printmsg(f"\n*TEST*evaluation results for epoch {str(state.epoch)} \n {report}", True, f"{logs_dir}/{run_name}.txt")
                 
            scores, pat_scores_msg, _ = pat.get_patient_level_metrics(pred_df, results_dir +f"test-epoch_wise/{run_name}/", f"{logs_dir}/{run_name}.txt", run_name+str(np.round(state.epoch)))
            printmsg("\n" + pat_scores_msg + "\n", True, f"{logs_dir}/{run_name}.txt")

            if scores['F1'] >= self.best_f1:
                self.best_f1 = scores['F1']
                self.best_epoch = int(np.round(state.epoch))

    generate_callback = GenerateTextCallback(trainer, test_dataset, test_df, valid_dataset)
    trainer.add_callback(generate_callback)

    printmsg(f"\n{datetime.now()} started training..", True, logfilepath)
    trainer.train()
    printmsg(f"\n{datetime.now()} completed training. Epoch wise performance results available at {logs_dir}/{run_name}.txt", True, logfilepath)

    printmsg(f"(For reference only, not for finetuning. Final model for inference is trained for {epochs} epochs.) best test F1 = {str(generate_callback.getbestmetric())} obtained at epoch {str(np.round(generate_callback.getbestepoch()))}", True, logfilepath) 
    return trainer


"""
Inference pipeline.
Output- predictions.csv file with gt and pred 
"""
def evaluate(data, df, model_trainerobj, sample_set, modelortrainer, results_dir, logfilepath, result_filename=""):
    data_loader = DataLoader(data, batch_size=8, shuffle=False)
    all_preds= []
    all_gts = []
    if modelortrainer == "model":
        for batch in data_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            logits = model_trainerobj(input_ids, attention_mask)
            probs = torch.sigmoid(torch.tensor(logits))            
            binary_preds = torch.round(probs)
            binary_preds = binary_preds.flatten().numpy()
            
            true_labels = batch['labels']
            true_labels = true_labels.flatten()

            all_preds.extend(binary_preds) 
            all_gts.extend(true_labels) 
            
    elif modelortrainer == "trainer":
        predictions = model_trainerobj.predict(data)
        # 'predictions' is a tuple, and the logits are stored in the first element of the tuple
        logits = predictions.predictions
        probs = torch.sigmoid(torch.tensor(logits))
        binary_preds = torch.round(probs)
        all_preds = binary_preds.flatten().numpy().tolist()
        logits_list = logits.flatten().tolist()
        probs_list = probs.flatten().numpy().tolist()
        
        true_labels = predictions.label_ids
        all_gts = true_labels.flatten()           
    
    gts = [int(x) for x in all_gts]
    preds = [int(x) for x in all_preds]
    rounds =  [x for x in df['round'].values]
    index_keys = [x for x in df['index_key'].values]
    cand_keys = [x for x in df['cand_key'].values]  

    pred_df = pd.DataFrame(list(zip(rounds, index_keys, cand_keys, gts, preds, probs_list, logits_list)), columns = ['round','index_key','cand_key','gt','pred', 'prob', 'logit'])
    if result_filename != "":
        pred_df.to_csv(results_dir+result_filename+".csv")

    report = classification_report(gts, preds, target_names=['Class 0', 'Class 1'])
    printmsg(f"\n------ {sample_set}: classification report ------ \n{report}", True, logfilepath)

    return pred_df


'''
get patient level metrics. Input is a results df from the training/inference pipeline that has a prediction for each candidate
'''
def get_patient_level_metrics(results, results_dir, logfilepath, resultsfilename=""):
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
        gt = results[results['unique_keys']==ikey]['gt'].values.tolist()
        tp_np_counter.loc[ikey, 'gt'] = str(gt)
        preds = results[results['unique_keys']==ikey]['pred'].values.tolist()
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
    
    
    printmsg(f"\nTP: {str(TP)} FP: {str(FP)} TN: {str(TN)} FN: {str(FN)}", True, logfilepath)
    if resultsfilename != "":
        tp_np_counter.to_csv(f"{results_dir}/{resultsfilename}_patientresults.csv")
    
    Precision = TP/(TP+FP) if TP+FP > 0 else 0    
    Recall = TP/(TP+FN) if TP+FN > 0 else 0    
    F1 = 2*Precision*Recall/(Precision + Recall) if Precision + Recall > 0 else 0    
    scores = {"P":Precision, "R":Recall, "F1": F1}    
    
    printmsg(f"\nP: {str(Precision)} R: {str(Recall)} F1: {str(F1)}", True, logfilepath)   
    return scores

########################################################################################################################################################################        
########################################################################################################################################################################        

if __name__ == "__main__":
    checkpoint = "allenai/longformer-base-4096" #"yikuan8/Clinical-Longformer"
    tokenizer = LongformerTokenizer.from_pretrained(checkpoint)

    #important: enable this flag when the hyper parameter tuning needs to be done with the dev set
    train_val_flag = False
    #enable this flag when the train(4 folds) and test (1 fold) is used for training and testing with the 1 set of chosen/best hyperparameters. 
    train_test_flag = True

    #provide location for the data file (csv).
    input_data = "data/full_data.csv"
    #output directory is used to store the model checkpoints, log files and predictions (results) in respective folders
    output_dir = "results/latestWOdev60_correct/"
    os.makedirs(output_dir, exist_ok= True)

    TRAIN_BATCHSIZE = [8, 16]
    VAL_BATCHSIZE = 16
    LEARNING_RATE = [5e-5, 1e-3]
    DEV_EPOCHS = 20
    EPOCHS = 20
    MAX_seq_len = 2048
    gradient_accumulation_steps=[8, 4, 1]

    # hyperparams_sets = []
    hyperparams_sets = [{"train_batch_size":16, "lr": 5e-5, "gradient_ac_steps": 4}]

    save_checkpoints_dir = output_dir + "checkpoints/"
    os.makedirs(save_checkpoints_dir, exist_ok= True)
    if train_val_flag:
        os.makedirs(save_checkpoints_dir+"eval/", exist_ok= True)
    os.makedirs(save_checkpoints_dir+"test/", exist_ok= True)

    results_dir = output_dir + "results/"
    os.makedirs(results_dir, exist_ok= True)
    if train_val_flag:
        os.makedirs(results_dir+"eval/", exist_ok= True)
    os.makedirs(results_dir+"test/", exist_ok= True)

    logs_dir = output_dir + "logs/"
    os.makedirs(logs_dir, exist_ok= True)
    if train_val_flag:
        os.makedirs(logs_dir+"eval/", exist_ok= True)
    os.makedirs(logs_dir+"test/", exist_ok= True)

    #Read the folds and corresponding index ids
    with open("data/crossvalfolds.pkl", "rb") as fh:
        crossvalfolds = pickle.load(fh)
    
    #Read the input data file that has all the data from train, val and test
    full_input_data = pd.read_csv(input_data)

    fold_ids = [0,1,2,3,4,0,1,2,3,4]
    for i in range(5): #for i in [0,1,2,3,4]:
        printmsg(f"\n\n**************************** RUNNING FOLD {str(i)} ********************************", True, results_dir+ f"metrics_log.txt")
        
        #reserves 3 folds for train, 1 for dev and 1 for test
        train_fold = fold_ids[i:i+3]
        dev_fold = fold_ids[i+3]
        test_fold = fold_ids[i+4]
    
        printmsg(f"\ntrain = {train_fold}, dev = {dev_fold}, test = {test_fold}", True, results_dir+ f"metrics_log.txt")
        train_index_keys = crossvalfolds["split"+str(train_fold[0])] + crossvalfolds["split"+str(train_fold[1])] + crossvalfolds["split"+str(train_fold[2])]    
        val_index_keys = crossvalfolds["split"+str(dev_fold)]   
        test_index_keys = crossvalfolds["split"+str(test_fold)]


        best_hp_set = {}
        if train_val_flag:
            best_eval_f1 = 0
            testepochs = 0
            train_input_data = full_input_data[full_input_data['index_key'].isin(train_index_keys)]
            #get label weights
            label_counter = Counter(train_input_data['label'].tolist())
            yt_freq = [label_counter[l] for l in range(max(label_counter)+1)]
            yt_weight = torch.tensor([len(train_input_data)/x for x in yt_freq ])
            
            train_encoded = tokenizer(train_input_data['input'].tolist(),return_tensors='pt', truncation=True, padding=True, max_length=MAX_seq_len)
            labels_train = torch.tensor(train_input_data['label'].tolist())
            train_dataset = CustomDataset(train_encoded['input_ids'], train_encoded['attention_mask'], labels_train)

            val_input_data = full_input_data[full_input_data['index_key'].isin(val_index_keys)]
            val_encoded = tokenizer(val_input_data['input'].tolist(),return_tensors='pt', truncation=True, padding=True, max_length=MAX_seq_len)    
            labels_valid = torch.tensor(val_input_data['label'].tolist())
            valid_dataset = CustomDataset(val_encoded['input_ids'], val_encoded['attention_mask'], labels_valid)

            printmsg(f"\nPatient distribution: train = {str(len(np.unique(train_input_data['index_key'])))}, dev = {str(len(np.unique(val_input_data['index_key'])))}", True, results_dir+ f"metrics_log.txt")
            printmsg(f"\nNumber of rows: train = {str(len(train_input_data['label'].tolist()))}, dev = {str(len(val_input_data['label'].tolist()))}", True, results_dir+ f"metrics_log.txt")
            printmsg(f"\nLabel distribution: train = {str(Counter(train_input_data['label'].tolist()))}, dev = {str(Counter(val_input_data['label'].tolist()))}", True, results_dir+ f"metrics_log.txt")

            model = BertBinaryClassifier(checkpoint, yt_weight)
            for hpset_id, hpset in enumerate(hyperparams_sets):
                folder_unique_name = f"fold_{str(i)}_batch_{str(hpset['train_batch_size'])}_gracc_{str(hpset['gradient_ac_steps'])}_lr_{str(hpset['lr'])}"
                
                printmsg(f"\n----- Running training and evaluation for following hyper params -----\n"+str(hpset), True, results_dir+ f"metrics_log.txt")
                try:
                    os.makedirs(save_checkpoints_dir + "eval/", exist_ok=True)
                    os.makedirs(f"{save_checkpoints_dir}/eval/{folder_unique_name}", exist_ok=True)
                    trainer, bestepoch, exp_eval_f1 = training_pipe(model, train_dataset, valid_dataset, val_input_data, f"{save_checkpoints_dir}/eval/{folder_unique_name}", 
                                                                    results_dir+ "metrics_log.txt", DEV_EPOCHS, hpset["lr"], hpset['train_batch_size'],VAL_BATCHSIZE, 
                                                                    logs_dir+"eval/", results_dir, hpset["gradient_ac_steps"], folder_unique_name)
                    
                    printmsg(f"\nbest f1 {exp_eval_f1} obtained at epoch {bestepoch}", True, results_dir+ f"metrics_log.txt")
                    # valid_results = evaluate(valid_dataset, val_input_data, trainer, "eval", "trainer", results_dir+"eval/", results_dir+ "metrics_log.txt", f"{folder_unique_name}_predictions")
                    # val_scores = get_patient_level_metrics(valid_results, results_dir+"eval/", results_dir+ "metrics_log.txt", f"{folder_unique_name}_patient_results")
                    if exp_eval_f1 >= best_eval_f1:
                        best_eval_f1 = exp_eval_f1
                        best_hp_set = hyperparams_sets[hpset_id]
                        testepochs = bestepoch

                except Exception as e:
                    printmsg(f"\n----- FAILED: Running training and evaluation for following hyper params -----\n {str(hpset)}", True, results_dir+ f"metrics_log.txt")
                    printmsg(str(e), True, results_dir+ f"metrics_log.txt")
            
            printmsg(f"\n----- Best F1 score of {str(best_eval_f1)} obtained for following hyper params -----\n {str(best_hp_set)} epochs {testepochs}", True, results_dir+ f"metrics_log.txt")

        if train_test_flag:
            if train_val_flag == False:
                valid_dataset = None
                printmsg(f"\n*NOTE: Skipped validation using Dev set*", True, results_dir+ f"metrics_log.txt")
                if len(best_hp_set) == 0:
                    best_hp_set = hyperparams_sets[0]
            else:
                printmsg(f"\n----- Validation complete. Now merging the train and dev sets for final training -----", True, results_dir+ f"metrics_log.txt")
            printmsg(f"\n----- Training and Testing with the following hyper params -----\n {str(best_hp_set)}", True, results_dir+ f"metrics_log.txt")
            
            train_index_keys.extend(val_index_keys)
            train_input_data = full_input_data[full_input_data['index_key'].isin(train_index_keys)]

            #get label weights
            label_counter = Counter(train_input_data['label'].tolist())
            yt_freq = [label_counter[l] for l in range(max(label_counter)+1)]
            yt_weight = torch.tensor([len(train_input_data)/x for x in yt_freq ])

            train_encoded = tokenizer(train_input_data['input'].tolist(),return_tensors='pt', truncation=True, padding=True, max_length=MAX_seq_len)
            labels_train = torch.tensor(train_input_data['label'].tolist())
            train_dataset = CustomDataset(train_encoded['input_ids'], train_encoded['attention_mask'], labels_train)

            test_input_data = full_input_data[full_input_data['index_key'].isin(test_index_keys)] 
            test_encoded = tokenizer(test_input_data['input'].tolist(),return_tensors='pt', truncation=True, padding=True, max_length=MAX_seq_len)    
            labels_test = torch.tensor(test_input_data['label'].tolist())
            test_dataset = CustomDataset(test_encoded['input_ids'], test_encoded['attention_mask'], labels_test)
            
            printmsg(f"\nPatient distribution: train = {str(len(np.unique(train_input_data['index_key'])))}, test = {str(len(np.unique(test_input_data['index_key'])))}", True, results_dir+ f"metrics_log.txt")
            printmsg(f"\nNumber of rows: train = {str(len(train_input_data['label'].tolist()))}, test = {str(len(test_input_data['label'].tolist()))}", True, results_dir+ f"metrics_log.txt")            
            printmsg(f"\nLabel distribution for final training for the fold: train = {str(Counter(train_input_data['label'].tolist()))}, test = {str(Counter(test_input_data['label'].tolist()))}", True, results_dir+ f"metrics_log.txt")

            model = BertBinaryClassifier(checkpoint, yt_weight)

            folder_unique_name = f"fold_{str(i)}_batch_{str(best_hp_set['train_batch_size'])}_gracc_{str(best_hp_set['gradient_ac_steps'])}_lr_{str(best_hp_set['lr'])}"
            os.makedirs(save_checkpoints_dir + "test/", exist_ok=True)
            os.makedirs(f"{save_checkpoints_dir}/test/{folder_unique_name}", exist_ok=True)

            printmsg(f"\n----- Started full training with the best hyper params ------", True, results_dir+ f"metrics_log.txt")
            if valid_dataset:
                trainer = final_training_pipe(model, train_dataset, test_dataset, test_input_data, valid_dataset, 
                                            f"{save_checkpoints_dir}/test/{folder_unique_name}", results_dir+ "metrics_log.txt", EPOCHS, best_hp_set["lr"], best_hp_set['train_batch_size'], 
                                            logs_dir+"test/", results_dir, best_hp_set["gradient_ac_steps"], folder_unique_name)
            else:
                trainer = final_training_pipe(model, train_dataset, test_dataset, test_input_data, train_dataset, 
                                            f"{save_checkpoints_dir}/test/{folder_unique_name}", results_dir+ "metrics_log.txt", EPOCHS, best_hp_set["lr"], best_hp_set['train_batch_size'], 
                                            logs_dir+"test/", results_dir, best_hp_set["gradient_ac_steps"], folder_unique_name)
            printmsg(f"\n----- Started testing ------", True, results_dir+ f"metrics_log.txt")
            test_results = evaluate(test_dataset, test_input_data, trainer, "test", "trainer", results_dir+"test/", results_dir+ "metrics_log.txt", f"{folder_unique_name}_predictions")        
            # scores = get_patient_level_metrics(test_results, results_dir+"test/", results_dir+ "metrics_log.txt", f"{folder_unique_name}_patient_results")
            scores, pat_scores_msg, _ = pat.get_patient_level_metrics(test_results, results_dir+"/test/", results_dir+ f"metrics_log.txt", f"{folder_unique_name}_patient_results")
            printmsg("\n"+ pat_scores_msg + "\n", True, results_dir+ f"metrics_log.txt")
            printmsg(f"\n{datetime.now()} Completed training and inference for the fold. \n", True, results_dir+ f"metrics_log.txt")
    