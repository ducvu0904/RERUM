from model import DragonNetBase, tarreg_loss, EarlyStopper, dragonnet_loss, tarreg_loss_dual, dragonnet_loss_dual

from metrics import auqc
import torch 
import numpy as np
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
        early_stop_metric='loss'  # 'loss' hoặc 'qini'
    ):
        self.model = DragonNetBase(input_dim,shared_hidden=shared_hidden, outcome_hidden=outcome_hidden)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.early_stop_metric = early_stop_metric
        
        # Chọn early stopper phù hợp
        self.early_stop = EarlyStopper(patience=10, min_delta=0)

    def fit(self, train_t_loader, train_c_loader, val_loader):
        print ("🔃🔃🔃Begin training Dragonnet🔃🔃🔃")
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss=0
            for (xt, tt, yt), (xc, tc, yc) in zip(train_t_loader, train_c_loader):
                    # Concatenate batches to process through shared layers together
                    x_batch = torch.cat([xt, xc], dim=0).to(self.device)
                    yt, yc = yt.to(self.device), yc.to(self.device)
                    
                    batch_size_t = xt.shape[0]
                    batch_size_c = xc.shape[0]
                    
                    self.optim.zero_grad()
                    
                    # Single forward pass through shared layers
                    y0_pred, y1_pred, t_pred, eps = self.model(x_batch)
                    
                    # Split predictions for treatment and control groups
                    y0_pred_t, y0_pred_c = y0_pred[:batch_size_t], y0_pred[batch_size_t:]
                    y1_pred_t, y1_pred_c = y1_pred[:batch_size_t], y1_pred[batch_size_t:]
                    t_pred_t, t_pred_c = t_pred[:batch_size_t], t_pred[batch_size_t:]
                    
                    # Use dual stream loss
                    loss = tarreg_loss_dual(yt, yc, t_pred_t, t_pred_c, 
                                           y0_pred_t, y1_pred_t, y0_pred_c, y1_pred_c,
                                           eps, alpha=self.alpha, beta=self.beta)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            

            val_metric = self.validate(val_loader)
            metric_name = "Val Loss"
                    
            if (epoch+1) % 1 == 0:
                    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/(len(train_t_loader)+len(train_c_loader)):.4f} | {metric_name}: {val_metric:.4f}")
            
            # Early stopping


            if self.early_stop.early_stop(val_metric, epoch, model=self.model):
                    print(f"⏹️ Early stopped at epoch {epoch+1} because Val loss doesnt reduce.")
                    break
        
        # Khôi phục model về epoch tốt nhất
        self.early_stop.restore_best_model(self.model)
    def validate(self, val_loader):
        self.model.eval()
        val_loss=0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                y0, y1, t_p, eps = self.model(x)

                val_loss += tarreg_loss(y, t, t_p, y0, y1, eps, alpha=self.alpha, beta=self.beta).item()
        return val_loss / len(val_loader)
    
        
    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(x)
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