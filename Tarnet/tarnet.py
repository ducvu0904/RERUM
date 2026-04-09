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
from ranking import uplift_ranking_loss, response_ranking_loss

class Tarnet:
    def __init__(
        self, 
        cate_dims,
        num_count,
        shared_hidden=200, 
        outcome_hidden=100, 
        epochs=30,
        learning_rate= 1e-3,
        weight_decay = 1e-5,
        early_stop_metric='qini',
        use_ema=True,
        ema_alpha=0.25,
        patience=20,
        early_stop_start_epoch=0,
        ranking_start_epoch = 0,
        shared_dropout = 0,
        outcome_dropout = 0,
        uplift_ranking = 0.0,
        response_ranking = 0.0,
        max_samples = 1000
    ):
        # Optuna/manual configs sometimes pass numeric hyperparameters as float (e.g. 451.0).
        # Linear layer dimensions and epoch counters must be integers.
        shared_hidden = int(shared_hidden)
        outcome_hidden = int(outcome_hidden)
        epochs = int(epochs)
        patience = int(patience)
        early_stop_start_epoch = int(early_stop_start_epoch)
        ranking_start_epoch = int(ranking_start_epoch)
        max_samples = int(max_samples)

        self.model = TarnetBase(cate_dims, num_count, shared_hidden=shared_hidden, outcome_hidden=outcome_hidden, shared_dropout=shared_dropout, outcome_dropout=outcome_dropout)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor = 0.5, patience = 10, min_lr = 1e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stop_metric = early_stop_metric      
        self.uplift_lambda = uplift_ranking
        self.rr_lambda = response_ranking     
        self.max_samples = max_samples  
        self.ranking_start_epoch = ranking_start_epoch
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
        print ("🔃🔃🔃Begin training Tarnet🔃🔃🔃")
        print (f"📊 Early Stop Metric: {self.early_stop_metric.upper()}")
        print (f"📊 Early Stop Start Epoch: {self.early_stop_start_epoch + 1}")
        # Only start selecting/recording the best epoch once training is considered stable.
        # This guarantees early random Qini spikes (e.g. first 20 epochs) are ignored.
        selection_start_epoch = max(0, self.early_stop_start_epoch, self.ranking_start_epoch)
        patience_start_epoch = max(self.early_stop_start_epoch, selection_start_epoch)
        if selection_start_epoch > 0:
            print (f"📊 Score Selection Start Epoch: {selection_start_epoch + 1} (ignore earlier epochs)")
        
        if self.early_stop_metric == 'ema_qini':
            print (f"📊 Strategy: Best EMA Qini (alpha={self.ema_alpha})")
            print (f"   Restore to epoch with highest smoothed (EMA) Qini score")
            print (f"   Patience: {self.patience} epochs")
        elif self.early_stop_metric == 'qini':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with best raw Qini score")
        elif self.early_stop_metric == 'loss':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with lowest validation loss")
            print (f"   Patience: {self.patience} epochs")
        # TRAINING LOOP
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss=0
            for x_cate, x_num, t_batch, y_batch in train_loader:
                    x_cate = x_cate.to(self.device)
                    x_num = x_num.to(self.device)
                    
                    t_batch =t_batch.to(self.device) 
                    y_batch = y_batch.to(self.device)
                    
                    t_mask = (t_batch.squeeze(1) == 1)
                    c_mask = (t_batch.squeeze(1) == 0)
                    self.optim.zero_grad()
                    
                    #FORWARD PASS
                    y0_pred, y1_pred = self.model(x_cate, x_num)
                    
                    y_t = y_batch[t_mask]
                    y_c = y_batch[c_mask]

                    y0_pred_c = y0_pred[c_mask]
                    y1_pred_t = y1_pred[t_mask]

                    base_loss = outcome_loss(y_t= y_t, y_c= y_c, y1_pred=y1_pred_t, y0_pred= y0_pred_c)
                    
                    if epoch >= self.ranking_start_epoch:
                        # Only compute uplift loss if lambda > 0 to avoid unnecessary computation
                        if self.uplift_lambda > 0 and self.rr_lambda  > 0:
                            uplift_loss = uplift_ranking_loss(y_true= y_batch, t_true=t_batch, y0_pred=y0_pred, y1_pred=y1_pred) * self.uplift_lambda
                            response_loss = response_ranking_loss(y_true = y_batch, t_true = t_batch, y0_pred = y0_pred, y1_pred = y1_pred, max_samples = self.max_samples) * self.rr_lambda
                            loss = base_loss + uplift_loss + response_loss
                        elif self.uplift_lambda > 0:
                            uplift_loss = uplift_ranking_loss(y_true= y_batch, t_true=t_batch, y0_pred=y0_pred, y1_pred=y1_pred) * self.uplift_lambda 
                            response_loss = torch.tensor(0.0)
                            loss = base_loss + uplift_loss
                        elif self.rr_lambda > 0: 
                            response_loss = response_ranking_loss(y_true = y_batch, t_true = t_batch, y0_pred = y0_pred, y1_pred = y1_pred, max_samples = self.max_samples) * self.rr_lambda
                            uplift_loss = torch.tensor(0.0)
                            loss = base_loss +  response_loss
                        else:
                            uplift_loss = torch.tensor(0.0, device=self.device)  
                            response_loss = torch.tensor(0.0, device=self.device)  
                            loss = base_loss
                    else:
                        loss = base_loss
                        uplift_loss = torch.tensor(0.0, device=self.device)  
                        response_loss = torch.tensor(0.0, device=self.device)  
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            
            # CALCULATE QINI AND LOSS
            val_qini = self.validate_qini(val_loader)
            val_loss = self.validate(val_loader, epoch)
            
            # Step the scheduler based on the selected early stopping metric
            if self.early_stop_metric == "loss":
                self.scheduler.step(val_loss)
            else: 
                self.scheduler.step(val_qini)
            current_lr = self.optim.param_groups[0]['lr']

            # Early stopping based on selected metric
            
            # EMA QINI EARLY STOP
            if self.early_stop_metric == 'ema_qini':
                if epoch >= selection_start_epoch:
                    # Update EMA only from selection_start_epoch onward so pre-ranking epochs are fully ignored.
                    if self.ema_qini is None:
                        self.ema_qini = val_qini
                    else:
                        self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini
                    
                    # Track best EMA Qini (patience only after patience_start_epoch)
                    if self.ema_qini > self.best_ema_qini:
                        self.best_ema_qini = self.ema_qini
                        self.best_ema_epoch = epoch
                        self.best_ema_model_state = copy.deepcopy(self.model.state_dict())
                        self.best_qini = val_qini  # Track raw qini at this epoch too
                        self.patience_counter = 0
                        best_marker = "⭐ NEW BEST EMA"
                    else:
                        if epoch >= patience_start_epoch:
                            self.patience_counter += 1
                        best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                else:
                    best_marker = "(ignored before score selection start epoch)"
                
                if (epoch+1) % 1 == 0:
                    uplift_info = f"Uplift Loss: {uplift_loss.item():.6f} | " if self.uplift_lambda > 0 else ""
                    pairwise_info = f"Response Loss: {response_loss.item():.6f} | " if self.rr_lambda > 0 else ""
                    ema_display = f"{self.ema_qini:.4f}" if self.ema_qini is not None else "N/A"
                    best_ema_display = f"{self.best_ema_qini:.4f}" if self.best_ema_model_state is not None else "N/A"
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Base Loss: {base_loss.item():.4f} | "
                        f"{uplift_info}"
                        f"{pairwise_info}"
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} | "
                        f"EMA Qini: {ema_display} | "
                        f"Best EMA: {best_ema_display} {best_marker}"
                    )
                
                # Early stopping based on patience
                if epoch >= patience_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in EMA Qini for {self.patience} epochs")
                    break
            
            # LOSS EARLY STOP
            elif self.early_stop_metric == 'loss':
                if epoch >= selection_start_epoch:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_qini = val_qini  # Track qini too for reporting
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.patience_counter = 0
                        best_marker = "⭐ NEW BEST (lowest loss)"
                    else:
                        if epoch >= patience_start_epoch:
                            self.patience_counter += 1
                        best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                else:
                    best_marker = "(ignored before score selection start epoch)"
                
                if (epoch+1) % 1 == 0:
                    uplift_info = f"Uplift Loss: {uplift_loss.item():.6f} | " if self.uplift_lambda > 0 else ""
                    pairwise_info = f"Response Loss: {response_loss.item():.6f} | " if self.rr_lambda > 0 else ""
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Base Loss: {base_loss.item():.4f} | "
                        f"{uplift_info}"
                        f"{pairwise_info}"
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} {best_marker}"
                        f" | LR: {current_lr:.4f}"
                    )
                
                # ONLY USE EARLYSTOP AFTER N EPOCHS
                if epoch >= patience_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in validation loss for {self.patience} epochs")
                    break
              
            # QINI EARLYSTOP    
            elif self.early_stop_metric == 'qini':
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

                if (epoch+1) % 1 == 0:
                    uplift_info = f"Uplift Loss: {uplift_loss.item():.6f} | " if self.uplift_lambda > 0 else ""
                    pairwise_info = f"Response Loss: {response_loss.item():.6f} | " if self.rr_lambda > 0 else ""
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Base Loss: {base_loss.item():.4f} | "
                        f"{uplift_info}"
                        f"{pairwise_info}"
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} {best_marker}"
                    )
        
        # RESTORE BEST MODEL
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
            else:
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1} with best Qini score: {self.best_qini:.4f}")
        else:
            print(f"\n⚠️ No valid model state saved. Using final epoch model.")
            if self.early_stop_metric == 'ema_qini':
                print(f"   Final EMA Qini: {self.ema_qini:.4f}" if self.ema_qini is not None else "   EMA not initialized")
            
    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss=0
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

                base_loss = outcome_loss(y_t= y_t, y_c= y_c, y1_pred=y1_pred_t, y0_pred= y0_pred_c)
                
                val_loss += base_loss.item()
        return val_loss / len(val_loader)
    
    def validate_qini(self, val_loader):
        self.model.eval()
        y_list = []
        t_list = []
        uplift_list = []
        
        with torch.no_grad():
            for x_cate, x_num, t, y in val_loader:
                x_cate = x_cate.to(self.device)
                x_num = x_num.to(self.device)
                y0_pred, y1_pred = self.model(x_cate, x_num)
                
                uplift = y1_pred - y0_pred
                
                y_list.append(y.to(self.device))
                t_list.append(t.to(self.device))
                uplift_list.append(uplift)
        
        # Concatenate all batches on GPU
        y_all = torch.cat(y_list, dim=0)
        t_all = torch.cat(t_list, dim=0)
        uplift_all = torch.cat(uplift_list, dim=0)
        
        # Convert to CPU numpy arrays because auqc uses numpy/pandas internally.
        qini_score = auqc(
            y_true=y_all.detach().cpu().numpy(),
            t_true=t_all.detach().cpu().numpy(),
            uplift_pred=uplift_all.detach().cpu().numpy(),
            bins=100, 
            plot=False
        )
        
        return float(qini_score)
        
    def predict(self, x_cate, x_num):
        self.model.eval()
        x_cate = x_cate.to(device=self.device, dtype=torch.long)
        x_num = x_num.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            y0_pred, y1_pred = self.model(x_cate, x_num)
        return y0_pred, y1_pred
            
# Xem uplift của một cá nhân bất kỳ
def get_individual_uplift(model, x_test_tensor, index):

    # Lấy features của cá nhân đó
    x_individual = x_test_tensor[index:index+1]  # Giữ shape [1, num_features]

    # Move the individual tensor to the same device as the model
    device = model.device # Get the device of the model using its stored attribute
    x_individual = x_individual.to(device)

    # Predict
    y0_pred, y1_pred = model.predict(x_individual)

    # Tính uplift
    uplift = (y1_pred - y0_pred).item()

    result = {
        "index": index,
        "y0_pred (control outcome)": y0_pred.item(),
        "y1_pred (treatment outcome)": y1_pred.item(),
        "uplift": uplift
    }

    return result         