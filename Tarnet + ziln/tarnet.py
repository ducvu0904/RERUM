from model import TarnetBase, EarlyStopper, outcome_loss, QiniEarlyStopper
from ziln import zero_inflated_lognormal_pred, compute_classification_metrics
try:
    from model import TarnetBase, EarlyStopper, outcome_loss, QiniEarlyStopper
except ModuleNotFoundError:
    from Tarnet.model import TarnetBase, EarlyStopper, outcome_loss, QiniEarlyStopper
import sys
from pathlib import Path
project_root = Path("/home/ducvu0904/Documents/Lab/RERUM")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from metrics import auqc
import torch 
import numpy as np
import copy



class Tarnet:
    def __init__(
        self, 
        cate_dims,
        num_count,
        shared_hidden=200, 
        outcome_hidden=100, 
        epochs=150,
        learning_rate= 1e-3,
        weight_decay = 1e-5,
        early_stop_metric='qini',
        use_ema=True,
        ema_alpha=0.15,
        patience=15,
        early_stop_start_epoch=0,
        outcome_dropout = 0,
        shared_dropout = 0,
        positive_rate = 0.01,
        ziln_lambda=1.0,
        pos_weight=1.0
    ):
        self.model = TarnetBase(cate_dims, num_count, shared_hidden=shared_hidden, outcome_hidden=outcome_hidden,
                                outcome_dropout=outcome_dropout, shared_dropout=shared_dropout,
                                positive_rate=positive_rate)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stop_metric = early_stop_metric
        self.ziln_lambda = ziln_lambda
        self.pos_weight = pos_weight

        # EMA parameters
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.patience = patience
        self.early_stop_start_epoch = early_stop_start_epoch
        
        # Tracking
        self.best_qini = -np.inf
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model_state = None
        
        # EMA tracking
        self.ema_qini = None
        self.best_ema_qini = -np.inf
        self.best_ema_epoch = 0
        self.best_ema_model_state = None
        self.patience_counter = 0

    def fit(self, train_loader, val_loader):
        print ("🔃🔃🔃Begin training Tarnet🔃🔃🔃")
        print (f"📊 Early Stop Metric: {self.early_stop_metric.upper()}")
        print (f"📊 Early Stop Start Epoch: {self.early_stop_start_epoch + 1}")
        selection_start_epoch = max(0, self.early_stop_start_epoch)
        if selection_start_epoch > 0:
            print (f"📊 Score Selection Start Epoch: {selection_start_epoch + 1} (ignore earlier epochs)")
        
        if self.early_stop_metric == 'ema_qini':
            print (f"📊 Strategy: Best EMA Qini (alpha={self.ema_alpha})")
            print (f"   Restore to epoch with highest smoothed (EMA) Qini score")
            print (f"   Patience: {self.patience} epochs")
        elif self.early_stop_metric == 'qini' and self.use_ema:
            print (f"📊 Strategy: Two-Stage EMA Filter (alpha={self.ema_alpha})")
            print (f"   EMA filters noise spikes, Raw Qini determines peak height")
            print (f"   Select checkpoint: raw_qini is highest AND raw_qini >= ema_qini")
        elif self.early_stop_metric == 'qini':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with best raw Qini score")
        elif self.early_stop_metric == 'loss':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with lowest validation loss")
            print (f"   Patience: {self.patience} epochs")
        
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss = 0
            for x_cate, x_num, t_batch, y_batch in train_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                t_batch = t_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                t_mask = (t_batch.squeeze(1) == 1)
                c_mask = (t_batch.squeeze(1) == 0)
                self.optim.zero_grad()
                
                y0_pred, y1_pred = self.model(x_cate, x_num)
                
                y_t = y_batch[t_mask]
                y_c = y_batch[c_mask]
                y0_pred_c = y0_pred[c_mask]
                y1_pred_t = y1_pred[t_mask]

                loss, cls_loss, reg_loss, mu_mean_t, sigma_mean_t, mu_mean_c, sigma_mean_c = outcome_loss(y_t=y_t, y_c=y_c, y1_pred=y1_pred_t, y0_pred=y0_pred_c, ziln_lambda=self.ziln_lambda, pos_weight=self.pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                epoch_loss += loss.item()
            
            val_qini = self.validate_qini(val_loader)
            val_loss = self.validate(val_loader)
            cls_metrics = self.validate_classification_metrics(val_loader)
            
            # Format classification metrics string
            def fmt_metric(v):
                return f"{v:.4f}" if v is not None else "N/A"
            cls_str = (
                f"F1_c: {fmt_metric(cls_metrics['f1_c'])} | "
                f"PR_AUC_c: {fmt_metric(cls_metrics['pr_auc_c'])} | "
                f"F1_t: {fmt_metric(cls_metrics['f1_t'])} | "
                f"PR_AUC_t: {fmt_metric(cls_metrics['pr_auc_t'])}"
            )
            
            # EMA QINI EARLY STOP
            if self.early_stop_metric == 'ema_qini':
                if epoch >= selection_start_epoch:
                    # Start EMA and best-checkpoint tracking only after selection_start_epoch.
                    if self.ema_qini is None:
                        self.ema_qini = val_qini
                    else:
                        self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini

                    if self.ema_qini > self.best_ema_qini:
                        self.best_ema_qini = self.ema_qini
                        self.best_ema_epoch = epoch
                        self.best_ema_model_state = copy.deepcopy(self.model.state_dict())
                        self.best_qini = val_qini
                        self.patience_counter = 0
                        best_marker = "⭐ NEW BEST EMA"
                    else:
                        self.patience_counter += 1
                        best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                else:
                    best_marker = "(ignored before score selection start epoch)"

                ema_display = f"{self.ema_qini:.4f}" if self.ema_qini is not None else "N/A"
                best_ema_display = f"{self.best_ema_qini:.4f}" if self.best_ema_model_state is not None else "N/A"
                
                print(
                    f"Epoch {epoch+1}/{self.epoch} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Cls: {cls_loss:.4f} | Reg: {reg_loss:.4f} | "
                    f"mu_t: {mu_mean_t:.4f} | sigma_t: {sigma_mean_t:.4f} | "
                    f"mu_c: {mu_mean_c:.4f} | sigma_c: {sigma_mean_c:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"{cls_str} | "
                    f"Val Qini: {val_qini:.4f} | "
                    f"EMA Qini: {ema_display} | "
                    f"Best EMA: {best_ema_display} {best_marker}"
                )
                
                if epoch >= selection_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in EMA Qini for {self.patience} epochs")
                    break
            
            # LOSS EARLY STOP
            elif self.early_stop_metric == 'loss':
                if epoch >= selection_start_epoch:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_qini = val_qini
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.patience_counter = 0
                        best_marker = "⭐ NEW BEST (lowest loss)"
                    else:
                        self.patience_counter += 1
                        best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                else:
                    best_marker = "(ignored before score selection start epoch)"
                
                print(
                    f"Epoch {epoch+1}/{self.epoch} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Cls: {cls_loss:.4f} | Reg: {reg_loss:.4f} | "
                    f"mu_t: {mu_mean_t:.4f} | sigma_t: {sigma_mean_t:.4f} | "
                    f"mu_c: {mu_mean_c:.4f} | sigma_c: {sigma_mean_c:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"{cls_str} | "
                    f"Val Qini: {val_qini:.4f} {best_marker}"
                )
                
                if epoch >= selection_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in validation loss for {self.patience} epochs")
                    break
                    
            elif self.early_stop_metric == 'qini':
                if self.use_ema:
                    if epoch >= selection_start_epoch:
                        if self.ema_qini is None:
                            self.ema_qini = val_qini
                        else:
                            self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini

                        is_above_trend = val_qini >= self.ema_qini
                        is_new_peak = val_qini > self.best_qini

                        if is_new_peak and is_above_trend:
                            self.best_qini = val_qini
                            self.best_epoch = epoch
                            self.best_model_state = copy.deepcopy(self.model.state_dict())
                            self.patience_counter = 0
                            best_marker = "⭐ NEW BEST (peak ≥ trend)"
                        elif is_new_peak and not is_above_trend:
                            self.patience_counter += 1
                            best_marker = f"❌ peak below trend (patience: {self.patience_counter}/{self.patience})"
                        elif not is_new_peak and is_above_trend:
                            self.patience_counter += 1
                            best_marker = f"✓ above trend but not peak (patience: {self.patience_counter}/{self.patience})"
                        else:
                            self.patience_counter += 1
                            best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                    else:
                        best_marker = "(ignored before score selection start epoch)"

                    ema_trend_display = f"{self.ema_qini:.4f}" if self.ema_qini is not None else "N/A"
                    
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Cls: {cls_loss:.4f} | Reg: {reg_loss:.4f} | "
                        f"mu_t: {mu_mean_t:.4f} | sigma_t: {sigma_mean_t:.4f} | "
                        f"mu_c: {mu_mean_c:.4f} | sigma_c: {sigma_mean_c:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"{cls_str} | "
                        f"Raw Qini: {val_qini:.4f} | "
                        f"EMA Trend: {ema_trend_display} | "
                        f"{best_marker}"
                    )
                    
                    if epoch >= selection_start_epoch and self.patience_counter >= self.patience:
                        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                        print(f"   No valid peak (raw ≥ trend) found in last {self.patience} epochs")
                        break
                else:
                    if epoch >= selection_start_epoch:
                        if val_qini > self.best_qini:
                            self.best_qini = val_qini
                            self.best_epoch = epoch
                            self.best_model_state = copy.deepcopy(self.model.state_dict())
                            best_marker = "⭐ NEW BEST"
                        else:
                            best_marker = ""
                    else:
                        best_marker = "(ignored before score selection start epoch)"
                        
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Cls: {cls_loss:.4f} | Reg: {reg_loss:.4f} | "
                        f"mu_t: {mu_mean_t:.4f} | sigma_t: {sigma_mean_t:.4f} | "
                        f"mu_c: {mu_mean_c:.4f} | sigma_c: {sigma_mean_c:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"{cls_str} | "
                        f"Val Qini: {val_qini:.4f} {best_marker}"
                    )
        
        if self.early_stop_metric == 'ema_qini' and self.best_ema_model_state is not None:
            self.model.load_state_dict(self.best_ema_model_state)
            print(f"\n✅ Training completed! Restored model to epoch {self.best_ema_epoch+1}")
            print(f"   Best EMA Qini: {self.best_ema_qini:.4f}")
            print(f"   Raw Qini at best EMA epoch: {self.best_qini:.4f}")
            print(f"   Strategy: Selected epoch with highest smoothed (EMA) Qini")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.early_stop_metric == 'loss':
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1}")
                print(f"   Best Val Loss: {self.best_loss:.4f}")
                print(f"   Qini at best epoch: {self.best_qini:.4f}")
            elif self.early_stop_metric == 'qini' and self.use_ema:
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1}")
                print(f"   Best Raw Qini: {self.best_qini:.4f}")
                print(f"   Final EMA Trend: {self.ema_qini:.4f}")
                print(f"   Strategy: Selected highest peak that stayed above EMA trend")
            else:
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1} with best Qini score: {self.best_qini:.4f}")
        else:
            print(f"\n⚠️ No valid model state saved. Using final epoch model.")
            if self.early_stop_metric == 'ema_qini':
                print(f"   Final EMA Qini: {self.ema_qini:.4f}" if self.ema_qini is not None else "   EMA not initialized")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cate, x_num, t, y in val_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                t = t.to(self.device)
                y = y.to(self.device)
                t_mask = (t.squeeze(1) == 1)
                c_mask = (t.squeeze(1) == 0)

                y0, y1 = self.model(x_cate, x_num)
                y_t = y[t_mask]
                y_c = y[c_mask]
                y0_pred_c = y0[c_mask]
                y1_pred_t = y1[t_mask]

                val_loss_batch, _, _, _, _, _, _ = outcome_loss(y_t=y_t, y_c=y_c, y1_pred=y1_pred_t, y0_pred=y0_pred_c, ziln_lambda=1.0)
                val_loss += val_loss_batch.item()
        return val_loss / len(val_loader)
    
    def validate_qini(self, val_loader):
        """Calculate Qini coefficient on validation set"""
        self.model.eval()
        y_true_list = []
        t_true_list = []
        uplift_list = []
        
        with torch.no_grad():
            for x_cate, x_num, t, y in val_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                y0_pred, y1_pred = self.model(x_cate, x_num)

                # Convert ZILN predictions to expected values
                y0_pred = zero_inflated_lognormal_pred(y0_pred)
                y1_pred = zero_inflated_lognormal_pred(y1_pred)
                
                uplift = (y1_pred - y0_pred).cpu().numpy()
                
                y_true_list.extend(y.cpu().numpy())
                t_true_list.extend(t.cpu().numpy())
                uplift_list.extend(uplift)
        
        qini_score = auqc(
            y_true=np.array(y_true_list),
            t_true=np.array(t_true_list),
            uplift_pred=np.array(uplift_list),
            bins=100,
            plot=False
        )
        
        return qini_score
    
    def validate_classification_metrics(self, val_loader):
        """Calculate F1 and PR-AUC for classification component on validation set.
        
        Returns:
            dict with f1_c, pr_auc_c (control), f1_t, pr_auc_t (treatment)
        """
        self.model.eval()
        
        # Collect logits and labels separately for control and treatment
        y0_logits_list = []
        y1_logits_list = []
        y_c_list = []  # labels for control group
        y_t_list = []  # labels for treatment group
        
        with torch.no_grad():
            for x_cate, x_num, t, y in val_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                t = t.to(self.device)
                y = y.to(self.device)
                t_mask = (t.squeeze(1) == 1)
                c_mask = (t.squeeze(1) == 0)
                
                y0_logits, y1_logits = self.model(x_cate, x_num)
                
                # Collect control group: y0 logits with control labels
                if c_mask.sum() > 0:
                    y0_logits_list.append(y0_logits[c_mask].cpu())
                    y_c_list.append(y[c_mask])
                
                # Collect treatment group: y1 logits with treatment labels
                if t_mask.sum() > 0:
                    y1_logits_list.append(y1_logits[t_mask].cpu())
                    y_t_list.append(y[t_mask])
        
        results = {}
        
        # Compute metrics for control group (y0)
        if y0_logits_list:
            y0_logits_all = torch.cat(y0_logits_list, dim=0)
            y_c_all = torch.cat(y_c_list, dim=0)
            metrics_c = compute_classification_metrics(y_c_all, y0_logits_all)
            results['f1_c'] = metrics_c['f1']
            results['pr_auc_c'] = metrics_c['pr_auc']
        else:
            results['f1_c'] = None
            results['pr_auc_c'] = None
        
        # Compute metrics for treatment group (y1)
        if y1_logits_list:
            y1_logits_all = torch.cat(y1_logits_list, dim=0)
            y_t_all = torch.cat(y_t_list, dim=0)
            metrics_t = compute_classification_metrics(y_t_all, y1_logits_all)
            results['f1_t'] = metrics_t['f1']
            results['pr_auc_t'] = metrics_t['pr_auc']
        else:
            results['f1_t'] = None
            results['pr_auc_t'] = None
        
        return results
        
    def predict(self, x_cate, x_num):
        self.model.eval()
        if isinstance(x_cate, torch.Tensor):
            x_cate = x_cate.to(device=self.device, dtype=torch.float32)
        else:
            x_cate = torch.as_tensor(x_cate, dtype=torch.float32, device=self.device)
        
        if isinstance(x_num, torch.Tensor):
            x_num = x_num.to(device=self.device, dtype=torch.float32)
        else:
            x_num = torch.as_tensor(x_num, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            y0_pred, y1_pred = self.model(x_cate, x_num)
            y0_pred = zero_inflated_lognormal_pred(y0_pred)
            y1_pred = zero_inflated_lognormal_pred(y1_pred)
        return y0_pred, y1_pred

    def print_val_predictions(self, val_loader):
        """Print y0_pred and y1_pred for the entire validation set."""
        self.model.eval()
        y0_list = []
        y1_list = []
        with torch.no_grad():
            for x_cate, x_num, t, y in val_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                y0_pred, y1_pred = self.model(x_cate, x_num)
                y0_pred = zero_inflated_lognormal_pred(y0_pred)
                y1_pred = zero_inflated_lognormal_pred(y1_pred)
                y0_list.append(y0_pred.cpu().numpy())
                y1_list.append(y1_pred.cpu().numpy())
        y0_all = np.concatenate(y0_list, axis=0)
        y1_all = np.concatenate(y1_list, axis=0)
        print("y0_pred (validation set):")
        print(y0_all)
        print("y1_pred (validation set):")
        print(y1_all)
        return y0_all, y1_all


def get_individual_uplift(model, x_test_tensor, index):
    x_individual = x_test_tensor[index:index+1]

    device = model.device
    x_individual = x_individual.to(device)

    y0_pred, y1_pred = model.predict(x_individual)

    uplift = (y1_pred - y0_pred).item()

    result = {
        "index": index,
        "y0_pred (control outcome)": y0_pred.item(),
        "y1_pred (treatment outcome)": y1_pred.item(),
        "uplift": uplift,
    }

    return result
