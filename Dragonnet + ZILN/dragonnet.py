from model import DragonNetBase, EarlyStopper, dragonnet_loss, QiniEarlyStopper, tarreg_loss
from ziln import zero_inflated_lognormal_pred
from metrics import auqc
import torch 
import numpy as np
import copy

class Dragonnet:
    def __init__(
        self, 
        input_dim,
        shared_hidden=200, 
        outcome_hidden=100, 
        alpha=1.0,
        beta=1.0,
        epochs=25,
        learning_rate= 1e-3,
        weight_decay = 1e-4,
        early_stop_metric='qini',
        rr_lambda = 1e-4,
        ur_lambda = 10,
        max_sample = 200,
        use_ema=True,
        ema_alpha=0.15,
        patience=10,
        early_stop_start_epoch=0,
        outcome_dropout = 0,
        shared_dropout = 0
    ):
        self.model = DragonNetBase(input_dim,shared_hidden=shared_hidden, outcome_hidden=outcome_hidden, outcome_drop=outcome_dropout, shared_drop=shared_dropout )
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.early_stop_metric = early_stop_metric
        self.rr_lambda = rr_lambda
        self.max_rr_samples = max_sample
        self.ur_lambda = ur_lambda
        
        # EMA parameters
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.patience = patience
        self.early_stop_start_epoch = early_stop_start_epoch
        
        # Tracking cho best model dựa trên Qini score
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
        print ("🔃🔃🔃Begin training Dragonnet🔃🔃🔃")
        print (f"📊 Early Stop Metric: {self.early_stop_metric.upper()}")
        print (f"📊 Early Stop Start Epoch: {self.early_stop_start_epoch + 1}")
        
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
            epoch_loss=0
            for x_batch , t_batch ,y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    
                    t_batch =t_batch.to(self.device) 
                    y_batch = y_batch.to(self.device)
                    
                    t_mask = (t_batch.squeeze(1) == 1)
                    c_mask = (t_batch.squeeze(1) == 0)
                    self.optim.zero_grad()
                    
                    y0_pred, y1_pred, t_pred, eps = self.model(x_batch)
                    
                    y_t = y_batch[t_mask]
                    y_c = y_batch[c_mask]

                    y0_pred_c = y0_pred[c_mask]
                    y1_pred_t = y1_pred[t_mask]

                    base_loss = dragonnet_loss(y_t= y_t, y_c= y_c, t_true=t_batch, t_pred = t_pred, y1_pred=y1_pred_t, y0_pred= y0_pred_c, eps= eps, alpha=self.alpha)
                    tarreg_reg = tarreg_loss(y_true= y_batch, t_true= t_batch , t_pred = t_pred, y0_pred=y0_pred, y1_pred=y1_pred, eps=eps, beta= self.beta)
                    loss = base_loss + tarreg_reg
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            
            # Tính Qini score trên validation set sau mỗi epoch
            val_qini = self.validate_qini(val_loader)
            val_loss = self.validate(val_loader)
            
            # Early stopping based on selected metric
            # EMA QINI EARLY STOP
            if self.early_stop_metric == 'ema_qini':
                # Update EMA
                if self.ema_qini is None:
                    self.ema_qini = val_qini
                else:
                    self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini
                
                # Track best EMA Qini (always track, patience only after early_stop_start_epoch)
                if self.ema_qini > self.best_ema_qini:
                    self.best_ema_qini = self.ema_qini
                    self.best_ema_epoch = epoch
                    self.best_ema_model_state = copy.deepcopy(self.model.state_dict())
                    self.best_qini = val_qini
                    self.patience_counter = 0
                    best_marker = "⭐ NEW BEST EMA"
                else:
                    if epoch >= self.early_stop_start_epoch:
                        self.patience_counter += 1
                    best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                
                if (epoch+1) % 1 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Base Loss: {base_loss.item():.4f} | "
                        f"Tarreg Loss: {tarreg_reg.item():.6f} | "
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} | "
                        f"EMA Qini: {self.ema_qini:.4f} | "
                        f"Best EMA: {self.best_ema_qini:.4f} {best_marker}"
                    )
                
                if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in EMA Qini for {self.patience} epochs")
                    break
            
            elif self.early_stop_metric == 'loss':
                # Early stopping dựa trên validation loss
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_qini = val_qini  # Track qini too for reporting
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                    best_marker = "⭐ NEW BEST (lowest loss)"
                else:
                    self.patience_counter += 1
                    best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                
                if (epoch+1) % 1 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Base Loss: {base_loss.item():.4f} | "
                        f"Tarreg Loss: {tarreg_reg.item():.6f} | "
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} {best_marker}"
                    )
                
                # ONLY USE EARLYSTOP AFTER N EPOCHS
                if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in validation loss for {self.patience} epochs")
                    break
                    
            elif self.early_stop_metric == 'qini':
                # Early stopping dựa trên Qini score
                if self.use_ema:
                    # Two-Stage Strategy: EMA as noise filter, raw Qini determines peak
                    
                    # Step 1: Update EMA trend (for filtering, not for selection)
                    if self.ema_qini is None:
                        self.ema_qini = val_qini
                    else:
                        self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini
                    
                    # Step 2: Select checkpoint only if:
                    #   - raw_qini is highest so far
                    #   - AND raw_qini >= ema_qini (filters noise spikes below trend)
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
                    
                    if (epoch+1) % 1 == 0:
                        print(
                            f"Epoch {epoch+1}/{self.epoch} | "
                            f"Base Loss: {base_loss.item():.4f} | "
                            f"Tarreg Loss: {tarreg_reg.item():.6f} | "
                            f"Total Loss: {loss.item():.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Raw Qini: {val_qini:.4f} | "
                            f"EMA Trend: {self.ema_qini:.4f} | "
                            f"{best_marker}"
                        )
                    
                    # Early stopping based on patience (only after early_stop_start_epoch)
                    if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                        print(f"   No valid peak (raw ≥ trend) found in last {self.patience} epochs")
                        break
                else:
                    # Original: track raw Qini only
                    if val_qini > self.best_qini:
                        self.best_qini = val_qini
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        best_marker = "⭐ NEW BEST"
                    else:
                        best_marker = ""
                        
                    if (epoch+1) % 1 == 0:
                        print(
                            f"Epoch {epoch+1}/{self.epoch} | "
                            f"Base Loss: {base_loss.item():.4f} | "
                            f"Tarreg Loss: {tarreg_reg.item():.6f} | "
                            f"Total Loss: {loss.item():.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val Qini: {val_qini:.4f} {best_marker}"
                        )
        
        # Khôi phục model về epoch tốt nhất
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
        val_loss=0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                t_mask = (t.squeeze(1) == 1)
                c_mask = (t.squeeze(1) == 0)
                
                y0, y1, t_pred, eps = self.model(x)
                y_t = y[t_mask]
                y_c = y[c_mask]

                y0_pred_t = y0[t_mask]
                y0_pred_c = y0[c_mask]
                y1_pred_t = y1[t_mask]
                y1_pred_c = y1[c_mask]

                val_loss += dragonnet_loss(y_t= y_t, y_c= y_c, t_true=t, t_pred = t_pred, y1_pred=y1_pred_t, y0_pred= y0_pred_c, eps= eps, alpha=self.alpha).item()
        return val_loss / len(val_loader)
    
    def validate_qini(self, val_loader):
        """Tính Qini coefficient trên validation set"""
        self.model.eval()
        y_true_list = []
        t_true_list = []
        uplift_list = []
        
        with torch.no_grad():
            for x, t, y in val_loader:
                x = x.to(self.device)
                y0_pred, y1_pred, t_pred, eps = self.model(x)
                
                # Chuyển đổi ZILN predictions
                y0_pred = zero_inflated_lognormal_pred(y0_pred)
                y1_pred = zero_inflated_lognormal_pred(y1_pred)
                
                # Tính uplift
                uplift = (y1_pred - y0_pred).cpu().numpy()
                
                y_true_list.extend(y.cpu().numpy())
                t_true_list.extend(t.cpu().numpy())
                uplift_list.extend(uplift)
        
        # Tính Qini score (không plot)
        qini_score = auqc(
            y_true=np.array(y_true_list),
            t_true=np.array(t_true_list),
            uplift_pred=np.array(uplift_list),
            bins=100,  # Giảm số bins để nhanh hơn
            plot=False
        )
        
        return qini_score
        
    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(x)
            y0_pred = zero_inflated_lognormal_pred(y0_pred)
            y1_pred = zero_inflated_lognormal_pred(y1_pred)
        return y0_pred, y1_pred, t_pred, eps
            
# Xem uplift của một cá nhân bất kỳ
def get_individual_uplift(model, x_test_tensor, index):

    # Lấy features của cá nhân đó
    x_individual = x_test_tensor[index:index+1]  # Giữ shape [1, num_features]

    # Move the individual tensor to the same device as the model
    device = model.device # Get the device of the model using its stored attribute
    x_individual = x_individual.to(device)

    # Predict
    y0_pred, y1_pred, t_pred, _ = model.predict(x_individual)

    # Tính uplift
    uplift = (y1_pred - y0_pred).item()

    result = {
        "index": index,
        "y0_pred (control outcome)": y0_pred.item(),
        "y1_pred (treatment outcome)": y1_pred.item(),
        "uplift": uplift,
        "t_pred (propensity)": t_pred.item()
    }

    return result         