"""
Kiểm tra tại sao CFR và Dragonnet khác nhau ngay từ epoch đầu
"""
import sys
import os
import torch
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed trước khi import models
seed = 42
seed_everything(seed)

# Import models
sys.path.insert(0, r"c:\Users\Lenovo\Documents\Rankability uplift modeling\cfr")
from model import CFRBase

sys.path.insert(0, r"c:\Users\Lenovo\Documents\Rankability uplift modeling\dragonnet")
os.chdir(r"c:\Users\Lenovo\Documents\Rankability uplift modeling\dragonnet")
from model import DragonNetBase

print("="*70)
print("🔍 KIỂM TRA TẠI SAO EPOCH ĐẦU TIÊN ĐÃ KHÁC")
print("="*70)

input_dim = 10
batch_size = 5

# Tạo fake data
x = torch.randn(batch_size, input_dim)
y_true = torch.randn(batch_size, 1)
t = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)

print(f"\n📊 Input shape: {x.shape}")
print(f"📊 Treatment: {t.squeeze().tolist()}")

# Reset seed để cả 2 models init giống nhau
seed_everything(seed)

print("\n" + "="*70)
print("🔵 CFRNet")
print("="*70)
cfr = CFRBase(input_dim=input_dim, shared_hidden=200, outcome_hidden=100)
print(f"Architecture: input({input_dim}) → 200 → 200 → 100")
print(f"              Y heads: 100 → 100 → 100 → 1")

# Forward pass
z_cfr, y1_cfr, y0_cfr = cfr(x)
print(f"\nShared representation shape: {z_cfr.shape}")
print(f"Y0 prediction: {y0_cfr[:3].squeeze().tolist()}")
print(f"Y1 prediction: {y1_cfr[:3].squeeze().tolist()}")

# Tính loss
t_mask = (t.squeeze() == 1)
c_mask = (t.squeeze() == 0)
y_t = y_true[t_mask]
y_c = y_true[c_mask]
y1_pred_t = y1_cfr[t_mask]
y0_pred_c = y0_cfr[c_mask]

if len(y_t) > 0 and len(y_c) > 0:
    loss_cfr = torch.mean((y1_pred_t - y_t)**2) + torch.mean((y0_pred_c - y_c)**2)
    print(f"\n📊 Initial Loss (epoch 0): {loss_cfr.item():.6f}")
else:
    print("\n⚠️ Not enough samples in both groups")

# Count parameters
cfr_params = sum(p.numel() for p in cfr.parameters())
print(f"📊 Total parameters: {cfr_params:,}")

# Reset seed lại giống hệt
seed_everything(seed)

print("\n" + "="*70)
print("🔴 DragonNet (với alpha=0, beta=0)")
print("="*70)
dragon = DragonNetBase(input_dim=input_dim, shared_hidden=200, outcome_hidden=100)
print(f"Architecture: input({input_dim}) → 200 → 200 → 200  ⚠️ KHÁC!")
print(f"              Y heads: 200 → 100 → 100 → 1  ⚠️ KHÁC!")
print(f"              + T head: 200 → 1")
print(f"              + Epsilon: 1 → 1")

# Forward pass
y0_dragon, y1_dragon, t_pred, eps = dragon(x)
print(f"\nShared representation shape: N/A (không return z)")
print(f"Y0 prediction: {y0_dragon[:3].squeeze().tolist()}")
print(f"Y1 prediction: {y1_dragon[:3].squeeze().tolist()}")
print(f"T prediction:  {t_pred[:3].squeeze().tolist()}")

# Tính loss (giống CFR, không dùng T head)
y1_pred_t = y1_dragon[t_mask]
y0_pred_c = y0_dragon[c_mask]

if len(y_t) > 0 and len(y_c) > 0:
    loss_dragon = torch.mean((y1_pred_t - y_t)**2) + torch.mean((y0_pred_c - y_c)**2)
    print(f"\n📊 Initial Loss (epoch 0): {loss_dragon.item():.6f}")
else:
    print("\n⚠️ Not enough samples in both groups")

# Count parameters
dragon_params = sum(p.numel() for p in dragon.parameters())
print(f"📊 Total parameters: {dragon_params:,}")

print("\n" + "="*70)
print("🔍 SO SÁNH")
print("="*70)
print(f"CFR Loss:     {loss_cfr.item():.6f}")
print(f"Dragon Loss:  {loss_dragon.item():.6f}")
print(f"Difference:   {abs(loss_cfr.item() - loss_dragon.item()):.6f}")
print(f"\nCFR params:    {cfr_params:,}")
print(f"Dragon params: {dragon_params:,}")
print(f"Extra params:  {dragon_params - cfr_params:,}")

# So sánh predictions
print("\n" + "="*70)
print("🔍 SO SÁNH PREDICTIONS (3 samples đầu)")
print("="*70)
print("\nY0 predictions:")
print(f"  CFR:    {y0_cfr[:3].squeeze().detach().numpy()}")
print(f"  Dragon: {y0_dragon[:3].squeeze().detach().numpy()}")
print(f"  Diff:   {(y0_cfr[:3] - y0_dragon[:3]).abs().squeeze().detach().numpy()}")

print("\nY1 predictions:")
print(f"  CFR:    {y1_cfr[:3].squeeze().detach().numpy()}")
print(f"  Dragon: {y1_dragon[:3].squeeze().detach().numpy()}")
print(f"  Diff:   {(y1_cfr[:3] - y1_dragon[:3]).abs().squeeze().detach().numpy()}")

print("\n" + "="*70)
print("❓ TẠI SAO KHÁC NGAY TỪ EPOCH ĐẦU?")
print("="*70)
print("""
1. ⚠️ SHARED LAYER OUTPUT SIZE KHÁC:
   CFR:    100 dims
   Dragon: 200 dims
   → Khác nhau cơ bản về architecture!

2. ⚠️ Y HEADS INPUT SIZE KHÁC:
   CFR:    Nhận 100 dims từ shared layer
   Dragon: Nhận 200 dims từ shared layer
   → Số parameters của Y heads khác!

3. ⚠️ RANDOM INITIALIZATION:
   - Dù dùng cùng seed, nhưng architecture khác
   - Số lượng weights khác → init sequence khác
   - Dragonnet có thêm T head & epsilon → thêm random draws

4. ⚠️ FORWARD PASS KHÁC:
   - CFR tính:    z(100) → y0, y1
   - Dragon tính: z(200) → y0, y1, t_pred, eps
   → Computation hoàn toàn khác!

✅ KẾT LUẬN:
   Dù set alpha=0, beta=0, hai models VẪN KHÁC về:
   - Kiến trúc (shared size)
   - Capacity (số parameters)
   - Initialization
   
   → Kết quả KHÁC NGAY TỪ EPOCH 0!
   → Đây KHÔNG PHẢI là TarNet như nhau!
""")

print("\n" + "="*70)
print("💡 ĐỂ GIỐNG NHAU CẦN:")
print("="*70)
print("""
1. Sửa CFR: outcome_hidden = shared_hidden = 200
   HOẶC
2. Sửa Dragonnet: shared output = 100 (thay vì 200)
   
3. Remove T head & epsilon khỏi Dragonnet hoàn toàn

4. Đảm bảo architecture HOÀN TOÀN GIỐNG NHAU
""")
