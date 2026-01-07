from model import DragonNetBase, tarreg_loss, EarlyStopper, dragonnet_loss, QiniEarlyStopper
from ziln import zero_inflated_lognormal_pred
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
        response_lambda = 1.0,
        uplift_lambda = 1.0,
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
        self.response_lambda = response_lambda
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
                    
                    self.optim.zero_grad()
                    
                    y0_pred, y1_pred, t_pred, eps = self.model(x_batch)
                    
                    if epoch <=5:
                        loss = tarreg_loss(y_batch, t_batch, t_pred, y0_pred, y1_pred, eps, alpha=self.alpha, beta=self.beta, response_lambda=0, uplift_lambda=0)
                    else: 
                        loss = tarreg_loss(y_batch, t_batch, t_pred, y0_pred, y1_pred, eps, alpha=self.alpha, beta=self.beta, response_lambda=self.response_lambda, uplift_lambda=self.uplift_lambda)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            
            # Tính metric cho early stopping
            if self.early_stop_metric == 'qini':
                val_metric = self.validate_qini(val_loader)
                metric_name = "Qini"
            else:
                val_metric = self.validate(val_loader, epoch)
                metric_name = "Val Loss"
                    
            if (epoch+1) % 1 == 0:
                    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | {metric_name}: {val_metric:.4f}")
            
            # Early stopping (only from epoch 7 onwards when full loss is used)
            if epoch >= 6:
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
    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss=0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                y0, y1, t_p, eps = self.model(x)
                if epoch <=5:
                    val_loss += tarreg_loss(y, t, t_p, y0, y1, eps, alpha=self.alpha, beta=self.beta, response_lambda=0, uplift_lambda=0).item()
                else:
                    val_loss += tarreg_loss(y, t, t_p, y0, y1, eps, alpha=self.alpha, beta=self.beta, response_lambda=self.response_lambda, uplift_lambda=self.uplift_lambda).item()

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
            bins=50,  # Giảm số bins để nhanh hơn
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