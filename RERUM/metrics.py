import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import kendalltau

def auuc(y_true, t_true, uplift_pred, bins=100, plot=True):
    """
    AUUC (Area Under uplift curve)
    
    Parameters:
    -----------
    y_true: spend
    t_true: treatment
    uplift_pred: uplift score predict
    bins: amount of buckets
    ------------
    Return
    -----------
    auuc
    """
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()
    
    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)
    
    #split into bucket
    data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    
    #create random baseline
    control_data = data.loc[data['t']==0]
    treatment_data = data.loc[data['t']==1]
    
    mean_control = control_data['y'].mean()
    mean_treatment = treatment_data["y"].mean()
    
    random_control = (np.random.rand(len(control_data)) -0.5)/ 10000 + mean_control
    random_treatment = (np.random.rand(len(treatment_data))-0.5) /10000 + mean_treatment
    
    data.loc[data['t']==0, 'random'] = random_control
    data.loc[data['t']==1, 'random'] = random_treatment
    
    #Calculate cumulative gain
    
    cumulative_gain = []
    cumulative_random =[]
    population =[]
    bucket_ids = sorted(data['bucket'].unique())
    
    for idx, bucket_id in enumerate(bucket_ids):
        cumulative_data = data.loc[data['bucket'] <= bucket_id]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_control==0 or n_treatment==0:
            print(f"Bucket {bucket_id}: Empty group, skip")
            continue
        
        #calculate mean outcome
        mean_y_control = control_group['y'].mean()
        mean_y_treatment = treatment_group['y'].mean()
        
        #AUUC formular
        uplift_gain = (mean_y_treatment - mean_y_control) * n_total
        
        mean_random_control = control_group['random'].mean()
        mean_random_treatment = treatment_group['random'].mean()
        random_gain = (mean_random_treatment - mean_random_control) *n_total
        
        cumulative_gain.append(uplift_gain)
        cumulative_random.append(random_gain)
        population.append(n_total)
        
        #force random baseline to meet model at endpoint
    if len(cumulative_random) >0:
        cumulative_random[-1] = cumulative_gain[-1]
    
    #normalize
    gap0 = cumulative_gain[-1]
    
    norm_factor = abs(gap0) if abs(gap0) > 1e-9 else 1.0
    
    cumulative_gains_norm = [x / norm_factor for x in cumulative_gain]
    cumulative_rand_norm = [x/ norm_factor for x in cumulative_random]
    
    #normalize x axis
    pop_max = max(population)
    pop_fraction = [p/pop_max for p in population]
    
    #add (0,0)
    x_curve = np.append(0, pop_fraction)
    y_curve = np.append(0, cumulative_gains_norm)
    y_rand = np.append(0, cumulative_rand_norm)
    
    #calcute auc using trapezoid rule
    auuc_score = np.trapezoid(y_curve, x_curve)
    auuc_rand = np.trapezoid(y_rand, x_curve)
    
    #visualize
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x_curve, y_curve, marker='o', markersize =4,
                 label = f"AUUC score = {auuc_score:.4f}", color= "darkgreen")
        plt.plot(x_curve, y_rand, marker='s', markersize=4,
                label=f'Random AUUC={auuc_rand:.4f})', 
                color='gray', linestyle='--', alpha=0.7)
        plt.xlabel("Cumulative percentage of people targeted")
        plt.ylabel("Cumulative uplift")
        plt.title("AUUC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return auuc_score

def auqc(y_true, t_true, uplift_pred, bins=100, plot=True):
    """
    AUQC (Area uplift under qini curve)
    
    Parameters:
    -----------
    y_true: spend
    t_true: treatment
    uplift_pred: uplift score predict
    bins: amount of buckets
    ------------
    Return
    -----------
    auqc
    """
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()

    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)
    
    #split into bucket
    data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    
    #create random baseline
    control_data = data.loc[data['t']==0]
    treatment_data = data.loc[data['t']==1]
    
    mean_control = control_data['y'].mean()
    mean_treatment = treatment_data["y"].mean()
    
    random_control = (np.random.rand(len(control_data)) -0.5)/ 10000 + mean_control
    random_treatment = (np.random.rand(len(treatment_data))-0.5) /10000 + mean_treatment
    
    data.loc[data['t']==0, 'random'] = random_control
    data.loc[data['t']==1, 'random'] = random_treatment
    
    #Calculate cumulative gain
    
    cumulative_gain = []
    cumulative_random =[]
    population =[]
    bucket_ids = sorted(data['bucket'].unique())
    
    for idx, bucket_id in enumerate(bucket_ids):
        cumulative_data = data.loc[data['bucket'] <= bucket_id]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_control==0 or n_treatment==0:
            print(f"Bucket {bucket_id}: Empty group, skip")
            continue
        
        #calculate mean outcome
        sum_y_control = control_group['y'].sum()
        sum_y_treatment = treatment_group['y'].sum()
        
        #AUUC formular
        qini_gain = sum_y_treatment - sum_y_control * (n_treatment/n_control)
        
        sum_random_control = control_group['random'].sum()
        sum_random_treatment = treatment_group['random'].sum()
        random_gain = sum_random_treatment - sum_random_control *(n_treatment/n_control)
        
        cumulative_gain.append(qini_gain)
        cumulative_random.append(random_gain)
        population.append(n_total)
        
        #force random baseline to meet model at endpoint
    if len(cumulative_random) >0:
        cumulative_random[-1] = cumulative_gain[-1]
    
    #normalize
    gap0 = cumulative_gain[-1]
    
    norm_factor = abs(gap0) if abs(gap0) > 1e-9 else 1.0
    
    cumulative_gains_norm = [x / norm_factor for x in cumulative_gain]
    cumulative_rand_norm = [x/ norm_factor for x in cumulative_random]
    
    #normalize x axis
    pop_max = max(population)
    pop_fraction = [p/pop_max for p in population]
    
    #add (0,0)
    x_curve = np.append(0, pop_fraction)
    y_curve = np.append(0, cumulative_gains_norm)
    y_rand = np.append(0, cumulative_rand_norm)
    
    #calcute auc using trapezoid rule
    qini_score = np.trapezoid(y_curve, x_curve)
    qini_rand = np.trapezoid(y_rand, x_curve)
    
    #visualize
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x_curve, y_curve, marker='o', markersize =4,
                 label = f"AUQC score = {qini_score:.4f}", color= "navy")
        plt.plot(x_curve, y_rand, marker='s', markersize=4,
                label=f'Random AUUC={qini_score:.4f})', 
                color='gray', linestyle='--', alpha=0.7)
        plt.xlabel("Cumulative percentage of people targeted")
        plt.ylabel("Cumulative qini")
        plt.title("AUQC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return qini_score

def lift (y_true, t_true, uplift_pred, h=0.3, bins=100, plot=True):
    """
    Lift@h 
    Parameters:
    -------------
    y_true: spend
    t_true: treatment (0/1)
    uplift_pred = uplift score
    h
    bins: amount of buckets
    -------------
    Return
    -------------
    Lift
    """
    
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()

    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)
    data['bucket'] = pd.qcut(-data['pred'], bins, labels= False, duplicates="drop")
    
    bucket_sorted = sorted(data['bucket'].unique())
    cutoff_idx = int(len(bucket_sorted) *h)
    cutoff_bucket = bucket_sorted[min(cutoff_idx, len(bucket_sorted)-1)]
    
    top_h_data = data.loc[data['bucket'] <= cutoff_bucket]
    
    n_total_top_h = len(top_h_data)
    
    control_top_h = top_h_data.loc[top_h_data['t']==0]
    treatment_top_h = top_h_data.loc[top_h_data['t']==1]
    
    mean_control = control_top_h['y'].mean()
    mean_treatment = treatment_top_h['y'].mean()
    
    lift_h = mean_treatment - mean_control
    
    return lift_h

def krcc(y_true, t_true, uplift_pred, bins=100):
    """
    KRCC (Kendall rank correlation coefficient)
    -----------
    Parameters
    -----------
    y_true: spend
    t_true: treament (0/1)
    uplift_pred: uplift score
    -----------
    Return
    -----------
    krcc
    """
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()
    
    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)
    data['bucket'] = pd.qcut(-data['pred'], bins, labels= False, duplicates="drop")
    
    cate_list = []
    pred_uplift_list =[]
    
    bucket_indices = sorted(data['bucket'].unique())
    for i in bucket_indices:
        dbucket = data.loc[data.bucket ==i]
        mean_control = dbucket.loc[dbucket['t']==0, 'y'].mean()
        mean_treatment = dbucket.loc[dbucket['t']==1, 'y'].mean()
        
        cate_val = mean_treatment - mean_control
        cate_list.append(cate_val)
        
        pred_val = dbucket['pred'].mean()
        pred_uplift_list.append(pred_val)
    
    pred_uplift_list_rank = np.argsort(pred_uplift_list)
    cate_list_rank = np.argsort(cate_list)
    
    correlation, _ = kendalltau(pred_uplift_list_rank, cate_list_rank)
    
    return correlation