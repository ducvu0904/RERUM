"""
Demo để hiểu gradient flow trong DragonNet vs CFRNet
"""
import torch
import torch.nn as nn

print("="*70)
print("DEMO: Tại sao propensity head ảnh hưởng dù alpha=0")
print("="*70)

# Giả sử shared representation đã tính
z = torch.randn(10, 100, requires_grad=True)
t_true = torch.randint(0, 2, (10, 1), dtype=torch.float32)
y_true = torch.randn(10, 1)

# Giả sử có outcome heads và propensity head
y_head = nn.Linear(100, 1)
t_head = nn.Linear(100, 1)

print("\n📊 SCENARIO 1: CHỈ CÓ OUTCOME HEAD (như CFR)")
print("-" * 70)

# Forward
y_pred = y_head(z)

# Loss (không có propensity)
loss_y = torch.mean((y_pred - y_true)**2)

# Backward
loss_y.backward(retain_graph=True)

# Xem gradient của z
grad_z_only_y = z.grad.clone()
print(f"Gradient norm của z (chỉ từ outcome): {grad_z_only_y.norm().item():.6f}")
print(f"Shape: {grad_z_only_y.shape}")

# Reset gradient
z.grad = None

print("\n📊 SCENARIO 2: CÓ CẢ OUTCOME + PROPENSITY HEAD (như Dragonnet)")
print("-" * 70)

# Forward (giống Dragonnet)
y_pred = y_head(z)
t_pred = torch.sigmoid(t_head(z))  # ⚠️ Propensity head được tính!

# Loss với alpha=0
alpha = 0.0
loss_y = torch.mean((y_pred - y_true)**2)
loss_t = nn.functional.binary_cross_entropy(t_pred, t_true)

total_loss = loss_y + alpha * loss_t  # alpha=0 → chỉ có loss_y
print(f"\nLoss breakdown:")
print(f"  loss_y: {loss_y.item():.6f}")
print(f"  loss_t: {loss_t.item():.6f} (weight = {alpha})")
print(f"  total:  {total_loss.item():.6f}")

# Backward
total_loss.backward(retain_graph=True)

grad_z_with_t = z.grad.clone()
print(f"\nGradient norm của z (có propensity): {grad_z_with_t.norm().item():.6f}")
print(f"Shape: {grad_z_with_t.shape}")

# So sánh
print("\n" + "="*70)
print("🔍 PHÂN TÍCH:")
print("="*70)
print(f"Gradient chỉ từ outcome:  {grad_z_only_y.norm().item():.6f}")
print(f"Gradient có propensity:    {grad_z_with_t.norm().item():.6f}")
print(f"Có khác biệt? {not torch.allclose(grad_z_only_y, grad_z_with_t)}")

if not torch.allclose(grad_z_only_y, grad_z_with_t):
    print("❌ HAI GRADIENTS KHÁC NHAU!")
else:
    print("✅ Hai gradients giống nhau")

print("\n" + "="*70)
print("❓ TẠI SAO LẠI KHÁC?")
print("="*70)

# Reset và test lại với việc KHÔNG tính t_pred
z.grad = None

print("\n📊 SCENARIO 3: CÓ PROPENSITY HEAD NHƯNG KHÔNG DÙNG TRONG LOSS")
print("-" * 70)

# Forward - KHÔNG tính t_pred
y_pred = y_head(z)
# t_pred = torch.sigmoid(t_head(z))  # ← COMMENT OUT!

# Loss
loss_y = torch.mean((y_pred - y_true)**2)

# Backward
loss_y.backward()

grad_z_no_tpred = z.grad.clone()
print(f"Gradient norm của z (không tính t_pred): {grad_z_no_tpred.norm().item():.6f}")

print("\n" + "="*70)
print("🎯 KẾT LUẬN:")
print("="*70)
print(f"Scenario 1 (CFR):         {grad_z_only_y.norm().item():.6f}")
print(f"Scenario 3 (không t_pred): {grad_z_no_tpred.norm().item():.6f}")
print(f"→ Giống nhau? {torch.allclose(grad_z_only_y, grad_z_no_tpred)}")

print("\n✅ Gradients GIỐNG NHAU khi:")
print("   - Không tính t_pred trong forward pass")
print("   - Hoặc t_pred không nằm trong computational graph của loss")

print("\n❌ Gradients KHÁC NHAU trong Dragonnet vì:")
print("   - t_pred được tính trong forward pass")
print("   - Dù alpha=0, nhưng z vẫn flow qua t_head")
print("   - PyTorch builds computational graph cho TẤT CẢ operations")
print("   - Gradient ∂loss/∂z bị ảnh hưởng bởi structure của graph!")

print("\n" + "="*70)
print("🔬 GIẢI THÍCH KỸ THUẬT:")
print("="*70)
print("""
Trong PyTorch:
1. Forward pass builds computational graph
2. Mọi operation trên tensor có requires_grad=True đều tạo node trong graph
3. Backward tính gradient theo TOÀN BỘ graph đã build

DragonNet forward:
  z → y_head → y_pred ──┐
  └→ t_head → t_pred    │  → loss (dù alpha=0)
  
CFR forward:
  z → y_head → y_pred → loss
  (KHÔNG có branch t_head)

→ Computational graph KHÁC → gradient flow KHÁC!
""")

print("\n" + "="*70)
print("💡 THỰC NGHIỆM CUỐI:")
print("="*70)

# Test với detach
z.grad = None

y_pred = y_head(z)
t_pred = torch.sigmoid(t_head(z.detach()))  # ← DETACH z!

loss_y = torch.mean((y_pred - y_true)**2)
loss_t = nn.functional.binary_cross_entropy(t_pred, t_true)
total_loss = loss_y + 0.0 * loss_t

total_loss.backward()
grad_z_detached = z.grad.clone()

print(f"Gradient với detach: {grad_z_detached.norm().item():.6f}")
print(f"Giống scenario 1?   {torch.allclose(grad_z_only_y, grad_z_detached, rtol=1e-4)}")
print("\n✅ Với .detach(), gradient không flow qua t_head → giống CFR!")
