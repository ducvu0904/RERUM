from model import DragonNetBase, tarreg_loss, EarlyStopper
from ziln import compute_expected_value
import torch 
import numpy as np
class Dragonnet:
    def __init__(
        self, 
        input_dim,
        ranking_lambda,
        shared_hidden=200, 
        outcome_hidden=100, 
        alpha=1.0,
        beta=1.0,
        epochs=25,
        learning_rate= 1e-3
    ):
        self.model = DragonNetBase(input_dim, shared_hidden, outcome_hidden)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.ranking_lambda = ranking_lambda
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
                    

                    loss = tarreg_loss(y_batch, t_batch, t_pred, y0_pred, y1_pred, eps, self.ranking_lambda, self.alpha, self.beta)
                    
                    loss.backward()
                    self.optim.step()
                    epoch_loss += loss.item()
                    
            val_loss = self.validate(val_loader)
            
            if (epoch+1) % 1 == 0:
                    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | VAL LOSS: {val_loss:.4f}")
            if self.early_stop.early_stop(val_loss):
                print(f"⏹️ Early stopped at epoch {epoch+1} because Val loss doesnt reduce.")
                break
    def validate(self, val_loader):
        self.model.eval()
        val_loss=0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                y0, y1, t_p, eps = self.model(x)

                val_loss += tarreg_loss(y, t, t_p, y0, y1, eps,self.ranking_lambda, self.alpha, self.beta).item()
        return val_loss / len(val_loader)
        
    def predict(self, x):
        self.model.eval()
        x = torch.Tensor(x)
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(x)
            y0_pred = compute_expected_value(y0_pred)
            y1_pred = compute_expected_value(y1_pred)
        return y0_pred, y1_pred, t_pred, eps
            
# Xem uplift của một cá nhân bất kỳ
def get_individual_uplift(model, x_test_tensor, index):
    """
    Lấy uplift prediction của một cá nhân theo index

    Parameters:
    -----------
    model: trained DragonNet model
    x_test_tensor: tensor chứa features của test set
    index: vị trí của cá nhân muốn xem (0, 1, 2, ...)

    Returns:
    --------
    Dictionary chứa thông tin uplift của cá nhân đó
    """
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