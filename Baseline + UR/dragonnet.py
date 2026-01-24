from model import DragonNetBase, EarlyStopper, dragonnet_loss, QiniEarlyStopper, tarreg_loss
from metrics import auqc
import torch 
import numpy as np
from ranking import uplift_ranking_loss
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
        early_stop_metric='loss',
        tarrreg_start_epoch = 200,
        uplift_lambda = 1
    ):
        self.model = DragonNetBase(input_dim,shared_hidden=shared_hidden, outcome_hidden=outcome_hidden)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.early_stop_metric = early_stop_metric
        self.tarreg_start_epoch = tarrreg_start_epoch
        self.uplift_lambda = uplift_lambda
        # Chọn early stopper phù hợp
        if early_stop_metric == 'qini':
            self.early_stop = QiniEarlyStopper(patience=10, min_delta=0)
        else:
            self.early_stop = EarlyStopper(patience=10, min_delta=0)

    def fit(self, train_loader, val_loader):
        print ("🔃🔃🔃Begin training Dragonnet🔃🔃🔃")
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
                    if epoch >= self.tarreg_start_epoch:
                        tarreg_reg = tarreg_loss(y_true= y_batch, t_true= t_batch , t_pred = t_pred, y0_pred=y0_pred, y1_pred=y1_pred, eps=eps, beta= self.beta)
                        uplift_loss = self.uplift_lambda * uplift_ranking_loss(y_true= y_batch, t_true = t_batch, y0_pred=y0_pred, y1_pred = y1_pred)
                        loss = base_loss + tarreg_reg + uplift_loss
                    else:
                        tarreg_reg = tarreg_loss(y_true= y_batch, t_true= t_batch , t_pred = t_pred, y0_pred=y0_pred, y1_pred=y1_pred, eps=eps, beta= 0)
                        uplift_loss = 0 * uplift_ranking_loss(y_true= y_batch, t_true = t_batch, y0_pred=y0_pred, y1_pred = y1_pred)
                        loss = base_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            
            # Tính metric cho early stopping
            if self.early_stop_metric == 'qini':
                val_metric = self.validate_qini(val_loader)
                metric_name = "Qini"
            else:
                val_metric = self.validate(val_loader)
                metric_name = "Val Loss"
                    
            if (epoch+1) % 5 == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Base Loss: {base_loss.item():.4f} | "
                    f"Tarreg Loss: {tarreg_reg.item():.6f} | "
                    f"Total Loss: {loss.item():.4f} | "\
                    f"Uplift loss: {uplift_loss:.4f} |"
                    f"Val Loss: {val_metric:.4f} |"

                )
            
            # Early stopping
            if self.early_stop_metric == 'qini':
                # Truyền model để lưu state khi có kết quả tốt nhất
                if self.early_stop.early_stop(val_metric, epoch, model=self.model):
                    print(f"⏹️ Early stopped at epoch {epoch+1}. Best Qini: {self.early_stop.best_qini:.4f} at epoch {self.early_stop.best_epoch+1}")
                    break
            else:
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