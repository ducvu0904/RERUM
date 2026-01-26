import torch
import torch.nn.functional as F
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
        
    def _ziln_mean(self, p, mu, sigma):
        """Tính Expected Value (LTV) dựa trên công thức ZILN"""
        # E[Y] = p * exp(mu + sigma^2 / 2)
        log_mean = mu + 0.5 * np.power(sigma, 2)
        # Clip log_mean để tránh tràn số khi exp
        log_mean = np.clip(log_mean, -10, 10) 
        return p * np.exp(log_mean)

    def _collect_data(self):
        """Thu thập dữ liệu và áp dụng công thức Sigma mới (Scaled Sigmoid)"""
        self.model.eval()
        
        y_trues, t_trues = [], []
        p_0s, mu_0s, sigma_0s, y0_preds = [], [], [], []
        p_1s, mu_1s, sigma_1s, y1_preds = [], [], [], []
        t_preds = []

        with torch.no_grad():
            for x, t, y in self.val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                
                # Forward pass
                raw_y0, raw_y1, t_p, eps_val = self.model(x)

                # --- CONTROL HEAD (T=0) ---
                # 1. Probability (Sigmoid)
                p0 = torch.sigmoid(raw_y0[:, 0]).cpu().numpy()
                
                # 2. Mu (Linear)
                mu0 = raw_y0[:, 1].cpu().numpy()
                
                # 3. Sigma (SCALED SIGMOID - CÔNG THỨC MỚI)
                # Range: [0.001, 2.001]
                sigma0 = (2.0 * torch.sigmoid(raw_y0[:, 2]) + 1e-3).cpu().numpy()
                
                # 4. Expected Value
                y0_ex = self._ziln_mean(p0, mu0, sigma0)

                # --- TREATMENT HEAD (T=1) ---
                p1 = torch.sigmoid(raw_y1[:, 0]).cpu().numpy()
                mu1 = raw_y1[:, 1].cpu().numpy()
                sigma1 = (2.0 * torch.sigmoid(raw_y1[:, 2]) + 1e-3).cpu().numpy()
                y1_ex = self._ziln_mean(p1, mu1, sigma1)

                # Lưu trữ
                y_trues.extend(y.cpu().numpy().flatten())
                t_trues.extend(t.cpu().numpy().flatten())
                t_preds.extend(torch.sigmoid(t_p).cpu().numpy().flatten())

                p_0s.extend(p0); mu_0s.extend(mu0); sigma_0s.extend(sigma0); y0_preds.extend(y0_ex)
                p_1s.extend(p1); mu_1s.extend(mu1); sigma_1s.extend(sigma1); y1_preds.extend(y1_ex)

        return pd.DataFrame({
            'y_true': y_trues, 't_true': t_trues, 't_pred': t_preds,
            'p_0': p_0s, 'mu_0': mu_0s, 'sigma_0': sigma_0s, 'y0_pred': y0_preds,
            'p_1': p_1s, 'mu_1': mu_1s, 'sigma_1': sigma_1s, 'y1_pred': y1_preds,
            'uplift': np.array(y1_preds) - np.array(y0_preds)
        })

    def plot_sanity_check(self):
        """Vẽ 4 biểu đồ quan trọng nhất cho Hillstrom + ZILN"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. P Distribution (Quan trọng: Phải tập trung quanh 0.01)
        sns.histplot(self.data['p_0'], color='blue', label='Control', ax=axes[0,0], kde=False, bins=50, alpha=0.5)
        sns.histplot(self.data['p_1'], color='orange', label='Treatment', ax=axes[0,0], kde=False, bins=50, alpha=0.5)
        axes[0,0].set_title('1. Probability of Purchase (Should be low ~0.01)')
        axes[0,0].set_xlim(0, 1) # Zoom vào vùng thấp vì Hillstrom conversion thấp
        axes[0,0].legend()

        # 2. Sigma Distribution (Quan trọng: Phải < 2.0 và có hình chuông)
        sns.histplot(self.data['sigma_0'], color='blue', label='Control', ax=axes[0,1], kde=True, bins=30, alpha=0.5)
        sns.histplot(self.data['sigma_1'], color='orange', label='Treatment', ax=axes[0,1], kde=True, bins=30, alpha=0.5)
        axes[0,1].set_title('2. Sigma Distribution (New Formula: Max 2.0)')
        axes[0,1].set_xlim(0, 2.1) # Giới hạn hiển thị đúng range sigmoid
        axes[0,1].legend()

        # 3. Mu vs Sigma (Quan trọng: Mu cao thì Sigma có cao không?)
        # Chỉ vẽ random 1000 điểm để đỡ lag
        subset = self.data.sample(min(1000, len(self.data)))
        axes[1,0].scatter(subset['mu_0'], subset['sigma_0'], alpha=0.3, label='Control', s=10, color='blue')
        axes[1,0].scatter(subset['mu_1'], subset['sigma_1'], alpha=0.3, label='Treatment', s=10, color='orange')
        axes[1,0].set_title('3. Correlation: Mu vs Sigma')
        axes[1,0].set_xlabel('Mu (Spend Magnitude)')
        axes[1,0].set_ylabel('Sigma (Uncertainty)')
        axes[1,0].legend()

        # 4. Uplift Distribution
        sns.histplot(self.data['uplift'], ax=axes[1,1], kde=True, color='green', bins=50)
        axes[1,1].set_title('4. Predicted Uplift (Treatment - Control)')
        axes[1,1].axvline(0, color='red', linestyle='--')

        plt.tight_layout()
        plt.show()

    def print_diagnostics(self):
        """In các thông số kỹ thuật"""
        print("🔍 --- DRAGONNET X-RAY REPORT ---")
        
        # 1. Check Epsilon (TarReg)
        try:
            eps_weight = self.model.epsilon.weight.item()
            print(f"🎯 TarReg Epsilon Value: {eps_weight:.6f}")
            if abs(eps_weight) < 1e-5:
                print("   -> CẢNH BÁO: Epsilon gần như bằng 0. TarReg chưa học hoặc Beta quá nhỏ.")
            else:
                print("   -> TỐT: Epsilon đã dịch chuyển, TarReg đang hoạt động.")
        except:
            print("   -> (Không tìm thấy layer epsilon)")

        # 2. Check Sigma Stats
        sig_mean = self.data[['sigma_0', 'sigma_1']].mean().mean()
        print(f"📉 Average Sigma: {sig_mean:.4f} (Lý tưởng: 0.5 - 1.5)")
        if sig_mean > 1.9:
            print("   -> CẢNH BÁO: Sigma bị bão hòa ở mức Max (2.0). Cần giảm learning rate.")
        
        # 3. Check Probability Stats
        p_mean_c = self.data['p_0'].mean()
        p_mean_t = self.data['p_1'].mean()
        print(f"📊 Avg Buy Probability - Control:   {p_mean_c:.2%}")
        print(f"📊 Avg Buy Probability - Treatment: {p_mean_t:.2%}")
        
        # Check Hillstrom Reality
        if p_mean_c > 0.10: # > 10%
             print("   -> CẢNH BÁO: P quá cao so với thực tế Hillstrom (~1%). Kiểm tra lại pos_weight.")

        print("✅ End Report.")