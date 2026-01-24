import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    try:
        data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    except:
        data['bucket'] = pd.cut(-data['pred'], bins, labels=False)
        
    # data = data.dropna(subset=['bucket'])
    if len(data) == 0:
        print("⚠️ All buckets are NaN after binning!")
        return 0.0 
    
    #create random baseline
    control_data = data.loc[data['t']==0.0]
    treatment_data = data.loc[data['t']==1.0]
    
    # print (control_data)
    # print (treatment_data)
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
    
    for i in bucket_ids:
        cumulative_data = data.loc[data['bucket'] <= i]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0.0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1.0]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_total ==0:
            continue
        if n_control==0 or n_treatment ==0:
            continue
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
        
    if len(cumulative_gain) == 0:
        print("⚠️ Warning: No valid buckets found. All buckets have empty treatment or control groups.")
        print(f"Treatment distribution: {(t_true == 1).sum()} treated, {(t_true == 0).sum()} control")
        return 0.0

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
        
    if gap0 < 0: 
        auuc_score, auuc_rand = np.trapezoid([i + 1 for i in np.append(0, cumulative_gains_norm)], np.append(0,(np.array(population))/max(population))),\
                                    np.trapezoid([i + 1 for i in np.append(0, cumulative_rand_norm)], np.append(0,(np.array(population))/max(population)))
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
    try:
        data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    except:
        data['bucket'] = pd.cut(-data['pred'], bins, labels=False)
    
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
    
    for bucket_id in bucket_ids:
        cumulative_data = data.loc[data['bucket'] <= bucket_id]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_control==0 or n_total==0:
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
        
    if len(cumulative_gain) == 0:
        print("⚠️ No valid buckets computed!")
        return 0.0
    
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
                label=f'Random AUQC={qini_rand:.4f})', 
                color='gray', linestyle='--', alpha=0.7)
        plt.xlabel("Cumulative percentage of people targeted")
        plt.ylabel("Cumulative qini")
        plt.title("AUQC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return qini_score

def lift (y_true, t_true, uplift_pred, h=0.3):
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
    
    y = np.array(y_true).flatten()
    t = np.array(t_true).flatten()
    pred = np.array(uplift_pred).flatten()
    df = pd.DataFrame({'y': y, 't': t, 'pred': pred})
    df = df.sort_values(by='pred', ascending=False).reset_index(drop=True)
    top_k = int(np.ceil(len(df) * h))
    top_df = df.iloc[:top_k]
    mean_c = top_df.loc[top_df['t']==0, 'y'].mean()
    mean_t = top_df.loc[top_df['t']==1, 'y'].mean()

    if np.isnan(mean_c) or np.isnan(mean_t):
        return np.nan
    return float(mean_t - mean_c)

def krcc(y_true, t_true, uplift_pred, bins=100):
    """
    KRCC (Kendall Rank Correlation Coefficient)
    
    Measures the rank correlation between predicted uplift scores and observed CATE.
    Implementation follows the standard approach in uplift modeling literature.
    
    Parameters:
    -----------
    y_true: spend (1d array)
    t_true: treatment (0/1) (1d array)  
    uplift_pred: predicted uplift score (1d array)
    bins: number of buckets to aggregate
    
    Returns:
    --------
    tau: Kendall's tau coefficient (float)
         Range: [-1, 1] where 1 = perfect agreement, -1 = perfect disagreement
    
    Reference:
    ----------
    Kendall's tau measures the ordinal association between two quantities.
    In uplift context, we want predicted uplift ranking to match true CATE ranking.
    """
    y = np.array(y_true).flatten()
    t = np.array(t_true).flatten()
    pred = np.array(uplift_pred).flatten()
    
    df = pd.DataFrame({'y': y, 't': t, 'pred': pred})
    df = df.sort_values(by='pred', ascending=False).reset_index(drop=True)

    try:
        df['bucket'] = pd.qcut(-df['pred'], bins, labels=False, duplicates='drop')
    except Exception:
        df['bucket'] = pd.cut(-df['pred'], bins, labels=False)

    pred_uplift_list = []
    cate_list = []

    bucket_indices = sorted(df['bucket'].dropna().unique())
    for b in bucket_indices:
        db = df[df['bucket'] == b]
        
        # Count samples in each group
        n_control = (db['t'] == 0).sum()
        n_treatment = (db['t'] == 1).sum()
        
        # Skip bins without both treatment and control
        if n_control == 0 or n_treatment == 0:
            continue

        mean_control = db.loc[db['t'] == 0, 'y'].mean()
        mean_treatment = db.loc[db['t'] == 1, 'y'].mean()

        if pd.isna(mean_control) or pd.isna(mean_treatment):
            continue

        # Observed CATE in this bin
        cate_val = float(mean_treatment - mean_control)
        # Average predicted uplift in this bin
        pred_val = float(db['pred'].mean())

        cate_list.append(cate_val)
        pred_uplift_list.append(pred_val)

    if len(cate_list) < 2:
        print(f"⚠️ Warning: Only {len(cate_list)} valid bins. Need at least 2 for correlation.")
        return np.nan  

    # Kendall's tau: correlation between rankings
    tau, p_value = kendalltau(pred_uplift_list, cate_list)

    return tau
