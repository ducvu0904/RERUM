import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import kendalltau

def get_uplift_metrics(y_true, t_true, uplift_pred, k=0.3, bins=100, plot=True):
    """
    Tính toán Metrics theo đúng chuẩn file utils.py của tác giả.
    Tách biệt logic tính AUUC và AUQC.
    """
    
    # --- 1. CHUẨN BỊ DỮ LIỆU ---
    # Ép kiểu an toàn
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()
    
    # Tạo DataFrame
    data = pd.DataFrame({'y': y_true, 't': t_true, 'pred': uplift_pred})
    
    # Sort theo điểm dự đoán giảm dần
    data = data.sort_values(by='pred', ascending=False)
    
    # Chia Bucket (qcut theo logic tác giả: -pred)
    try:
        data['bucket'] = pd.qcut(-data['pred'], bins, labels=False, duplicates='drop')
    except:
        data['bucket'] = pd.cut(-data['pred'], bins, labels=False) 

    # --- 2. TẠO RANDOM BASELINE (Theo utils.py) ---
    # Tác giả tạo feature 'random' bằng mean + nhiễu siêu nhỏ
    l0 = data.loc[data['t']==0]
    l1 = data.loc[data['t']==1]
    
    mean0 = l0['y'].mean()
    mean1 = l1['y'].mean()
    
    # Công thức nhiễu: (rand - 0.5)/100000 + mean
    r0 = (np.random.rand(len(l0)) - 0.5)/100000 + mean0
    r1 = (np.random.rand(len(l1)) - 0.5)/100000 + mean1
    
    data.loc[data['t']==0, 'random'] = r0
    data.loc[data['t']==1, 'random'] = r1
    
    # --- 3. HÀM TÍNH ĐƯỜNG CONG (Curve Calculator) ---
    def calculate_curve(metric_type):
        """
        metric_type: 'auuc' hoặc 'auqc'
        """
        res_list, pop_list, rand_res_list = [], [], []
        bucket_ids = sorted(data['bucket'].unique())
        
        for i in bucket_ids:
            # Lấy dữ liệu tích lũy
            dbucket = data.loc[data.bucket <= i]
            db_base = dbucket.loc[dbucket['t'] == 0] # Control
            db_exp = dbucket.loc[dbucket['t'] == 1]  # Treatment
            
            len_base = len(db_base)
            len_exp = len(db_exp)
            
            if len_base == 0 or len_exp == 0:
                continue
                
            # --- CÔNG THỨC KHÁC BIỆT ---
            if metric_type == 'auuc':
                # AUUC: Dùng Hiệu của Trung Bình * Tổng Dân Số
                # Formula: (Mean_T - Mean_C) * (N_T + N_C)
                val_gain = (db_exp['y'].mean() - db_base['y'].mean()) * (len_base + len_exp)
                rand_gain = (db_exp['random'].mean() - db_base['random'].mean()) * (len_base + len_exp)
            else: 
                # AUQC: Dùng Tổng T - Tổng C có scale
                # Formula: Sum_T - Sum_C * (N_T / N_C)
                val_gain = db_exp['y'].sum() - db_base['y'].sum() * (len_exp / len_base)
                rand_gain = db_exp['random'].sum() - db_base['random'].sum() * (len_exp / len_base)
            
            res_list.append(val_gain)
            rand_res_list.append(rand_gain)
            pop_list.append(len_base + len_exp)
            
        # Fix điểm cuối của Random bằng Model (ép buộc gặp nhau ở 100%)
        if len(res_list) > 0:
            rand_res_list[-1] = res_list[-1]
            
        return res_list, pop_list, rand_res_list

    # --- 4. TÍNH TOÁN & CHUẨN HÓA (Normalization) ---
    def normalize_and_score(res, pop, rand_res):
        if len(res) == 0: return 0.0, [], [], [], []
        
        # Gap0 là giá trị cuối cùng (Tổng Uplift toàn tập)
        gap0 = res[-1]
        
        # Chuẩn hóa chia cho abs(gap0) - Logic tác giả
        norm_factor = abs(gap0) if abs(gap0) > 1e-9 else 1.0
        
        y_norm = [x / norm_factor for x in res]
        y_rand_norm = [x / norm_factor for x in rand_res]
        
        # Trục hoành chuẩn hóa [0, 1]
        pop_max = max(pop)
        x_norm = [p / pop_max for p in pop]
        
        # Thêm điểm (0,0)
        y_final = np.append(0, y_norm)
        y_rand_final = np.append(0, y_rand_norm)
        x_final = np.append(0, x_norm)
        
        # Tính AUC bằng quy tắc hình thang (trapz)
        score = np.trapezoid(y_final, x_final)
        
        # Xử lý trường hợp gap0 âm (như trong code tác giả: if gap0 < 0 ...)
        # Tác giả cộng thêm diện tích hình vuông nếu gap < 0, 
        # nhưng ở đây ta cứ lấy raw score trước cho dễ so sánh.
        
        return score, x_final, y_final, y_rand_final

    # --- CHẠY TÍNH TOÁN ---
    # 1. AUUC
    res_u, pop_u, rand_u = calculate_curve('auuc')
    auuc_score, x_u, y_u, yr_u = normalize_and_score(res_u, pop_u, rand_u)
    
    # 2. AUQC
    res_q, pop_q, rand_q = calculate_curve('auqc')
    auqc_score, x_q, y_q, yr_q = normalize_and_score(res_q, pop_q, rand_q)

    # --- 5. TÍNH CÁC METRICS KHÁC ---
    # Lift@K
    h = k * len(set(data.bucket))
    dbucket = data.loc[data.bucket <= h]
    mean_t = dbucket.loc[dbucket['t']==1, 'y'].mean()
    mean_c = dbucket.loc[dbucket['t']==0, 'y'].mean()
    lift_val = mean_t - mean_c
    if np.isnan(lift_val): lift_val = 0.0
        
    # KRCC (Kendall Rank)
    bin_stats = data.groupby('bucket').apply(
        lambda x: x.loc[x['t']==1, 'y'].mean() - x.loc[x['t']==0, 'y'].mean()
    ).fillna(0)
    bin_preds = data.groupby('bucket')['pred'].mean().fillna(0)
    
    # argsort để lấy thứ hạng
    krcc, _ = kendalltau(np.argsort(bin_preds.values), np.argsort(bin_stats.values)) # type: ignore
    if np.isnan(krcc): krcc = 0.0 # type: ignore

    # --- 6. VẼ BIỂU ĐỒ (Dual Plot) ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot AUQC (Qini)
        axes[0].plot(x_q, y_q, marker='.', label=f'Model (AUQC={auqc_score:.4f})', color='navy')
        axes[0].plot(x_q, yr_q, marker='.', label='Random', color='gray', linestyle='--')
        axes[0].set_title('Qini Curve (AUQC)', fontsize=14)
        axes[0].set_xlabel('Fraction targeted')
        axes[0].set_ylabel('Normalized Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot AUUC (Uplift)
        axes[1].plot(x_u, y_u, marker='.', label=f'Model (AUUC={auuc_score:.4f})', color='darkgreen')
        axes[1].plot(x_u, yr_u, marker='.', label='Random', color='gray', linestyle='--')
        axes[1].set_title('Uplift Curve (AUUC)', fontsize=14)
        axes[1].set_xlabel('Fraction targeted')
        axes[1].set_ylabel('Normalized Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return {
        "AUQC": auqc_score,
        "AUUC": auuc_score,
        "Lift": lift_val,
        "KRCC": krcc
    }