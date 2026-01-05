import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DragonnetDebugger:
    def __init__(self, model, val_loader, device='cpu'):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.data = self._collect_data()

    def _collect_data(self):
        """Run the model on the validation loader and cache outputs."""
        self.model.eval()
        
        y_trues, t_trues = [], []
        y0_preds, y1_preds, t_preds = [], [], []

        with torch.no_grad():
            for x, t, y in self.val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                
                y0_pred, y1_pred, t_p, _ = self.model(x)

                # t_pred in DragonNetBase is already a probability, but we clamp to avoid 0/1.
                if torch.min(t_p) < 0 or torch.max(t_p) > 1:
                    t_p = torch.sigmoid(t_p)
                t_p = torch.clamp(t_p, 1e-4, 1 - 1e-4)

                y_trues.extend(y.cpu().numpy().flatten())
                t_trues.extend(t.cpu().numpy().flatten())
                t_preds.extend(t_p.cpu().numpy().flatten())
                y0_preds.extend(y0_pred.cpu().numpy().flatten())
                y1_preds.extend(y1_pred.cpu().numpy().flatten())

        # Đóng gói vào DataFrame cho dễ vẽ
        return pd.DataFrame({
            'y_true': y_trues, 't_true': t_trues, 't_pred': t_preds,
            'y0_pred': y0_preds, 'y1_pred': y1_preds,
            'uplift': np.array(y1_preds) - np.array(y0_preds)
        })

    def plot_all(self):
        """Vẽ toàn bộ biểu đồ chẩn đoán"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Propensity Score Distribution (Kiểm tra Positivity)
        sns.histplot(data=self.data, x='t_pred', hue='t_true', kde=True, ax=axes[0,0], bins=30, alpha=0.5)
        axes[0,0].set_title('Propensity Score Overlap (Cần chồng lấn tốt)')
        axes[0,0].set_xlabel('P(T=1|X)')
        
        # 2. Predicted outcomes distribution by arm
        sns.histplot(self.data['y0_pred'], color='blue', label='Y0 pred', ax=axes[0,1], kde=True, alpha=0.35)
        sns.histplot(self.data['y1_pred'], color='orange', label='Y1 pred', ax=axes[0,1], kde=True, alpha=0.35)
        axes[0,1].set_title('Predicted Outcomes (Y0 vs Y1)')
        axes[0,1].legend()

        # 3. Uplift Distribution
        sns.histplot(self.data['uplift'], ax=axes[1,0], kde=True, color='green')
        axes[1,0].set_title('Predicted Uplift Distribution')
        axes[1,0].axvline(0, color='red', linestyle='--')

        # 4. Calibration Check (Outcome thực tế vs Dự đoán)
        idx0 = self.data['t_true'] == 0
        idx1 = self.data['t_true'] == 1
        axes[1,1].scatter(self.data[idx0]['y_true'], self.data[idx0]['y0_pred'], alpha=0.3, s=10, label='Control')
        axes[1,1].scatter(self.data[idx1]['y_true'], self.data[idx1]['y1_pred'], alpha=0.3, s=10, label='Treatment', color='orange')
        max_val = max(self.data['y_true'].max(), self.data['y0_pred'].max(), self.data['y1_pred'].max())
        axes[1,1].plot([self.data['y_true'].min(), max_val], [self.data['y_true'].min(), max_val], 'r--')
        axes[1,1].set_title('Calibration: Y_True vs Y_Pred')
        axes[1,1].set_xlabel('Actual Outcome')
        axes[1,1].set_ylabel('Predicted Outcome')
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()

    def check_anomalies(self):
        """In ra các cảnh báo nếu phát hiện bất thường"""
        print("🔍 --- DIAGNOSTIC REPORT ---")
        
        # Check NaN
        n_nan = self.data.isna().sum().sum()
        if n_nan > 0:
            print(f"❌ CẢNH BÁO: Phát hiện {n_nan} giá trị NaN trong predictions!")
        
        # Check Propensity
        t_pred_mean = self.data['t_pred'].mean()
        if t_pred_mean < 0.05 or t_pred_mean > 0.95:
            print(f"⚠️ CẢNH BÁO: Propensity score bị lệch hẳn về 1 phía ({t_pred_mean:.2f}).")
            print("   -> Kiểm tra lại cân bằng mẫu hoặc hàm loss propensity.")

        # Check uplift sanity
        uplift_mean = self.data['uplift'].mean()
        if np.isnan(uplift_mean):
            print("❌ CẢNH BÁO: Uplift chứa NaN!")
        elif abs(uplift_mean) < 1e-5:
            print("ℹ️ Lưu ý: Uplift trung bình gần bằng 0. Kiểm tra lại xem mô hình có phân biệt được hai nhóm không.")

        print("✅ Diagnosis complete.")