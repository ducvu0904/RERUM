from model import DragonNetBase, tarreg_loss, EarlyStopper
import torch

class DragonNet:
    def __init__(self, input_dim, epochs=50, lr = 0.001, alpha =1.0, beta =1.0, ranking_lambda =1.0):
        self.model = DragonNetBase(input_dim)
        self.epochs = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.alpha = alpha
        self.beta = beta
        self.ranking_lambda = ranking_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit (self, train_loader, val_loader):
        print (f"Begin training Dragonnet Baseline🔃🔃🔃 ") 
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss=0
            for x_batch , t_batch ,y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                
                t_batch =t_batch.to(self.device) 
                y_batch = y_batch.to(self.device)
                
                self.optim.zero_grad()
                
                y0_pred, y1_pred, t_pred, eps = self.model(x_batch)
                
                loss = tarreg_loss(y_batch, t_batch, t_pred, y0_pred, y1_pred, eps, self.alpha, self.beta, self.ranking_lambda )
                
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
            
            val_loss = self.validate(val_loader)
            
            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | VAL LOSS: {val_loss:.4f}")
            
            # Early stopping with model checkpoint
            if early_stopper.early_stop(val_loss, epoch=epoch, model=self.model):
                print(f"⏹️ Early stop at epoch {epoch+1} ")
                break
        
        # Restore best model
        early_stopper.restore_best_model(self.model)
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, t, y in val_loader:
                X, t, y = X.to(self.device), t.to(self.device), y.to(self.device)
                y0, y1, t_p, eps = self.model(X)
                val_loss += tarreg_loss(y, t, t_p, y0, y1, eps, self.alpha, self.beta, self.ranking_lambda).item()
        return val_loss / len(val_loader)
                
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            y0, y1, t_pred, eps  = self.model(x.to(self.device))
        return y0, y1, t_pred, eps