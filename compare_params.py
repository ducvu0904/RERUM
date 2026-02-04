"""
So sánh số lượng parameters giữa CFR và Dragonnet
"""
import sys
import os

# Add paths
cfr_path = r"c:\Users\Lenovo\Documents\Rankability uplift modeling\cfr"
dragon_path = r"c:\Users\Lenovo\Documents\Rankability uplift modeling\dragonnet"

# Import CFR
sys.path.insert(0, cfr_path)
os.chdir(cfr_path)
from model import CFRBase

# Import Dragonnet
sys.path.insert(0, dragon_path)
os.chdir(dragon_path)
from model import DragonNetBase

# Giả sử input_dim = 10
input_dim = 10

cfr = CFRBase(input_dim=input_dim, shared_hidden=200, outcome_hidden=100)
dragon = DragonNetBase(input_dim=input_dim, shared_hidden=200, outcome_hidden=100)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("="*70)
print("📊 SO SÁNH SỐ LƯỢNG PARAMETERS")
print("="*70)

cfr_params = count_params(cfr)
dragon_params = count_params(dragon)

print(f"\nCFRNet:     {cfr_params:,} parameters")
print(f"DragonNet:  {dragon_params:,} parameters")
print(f"Difference: {dragon_params - cfr_params:,} parameters")
print(f"Ratio:      {dragon_params/cfr_params:.2f}x")

print("\n" + "="*70)
print("📋 BREAKDOWN:")
print("="*70)

print("\n🔵 CFRNet:")
print(f"  Shared layer: input({input_dim}) → 200 → 200 → 100")
print(f"  Y0 head:      100 → 100 → 100 → 1")
print(f"  Y1 head:      100 → 100 → 100 → 1")

print("\n🔴 DragonNet:")
print(f"  Shared layer: input({input_dim}) → 200 → 200 → 200")
print(f"  Y0 head:      200 → 100 → 100 → 1")
print(f"  Y1 head:      200 → 100 → 100 → 1")
print(f"  T head:       200 → 1  ⚠️ EXTRA")
print(f"  Epsilon:      1 → 1    ⚠️ EXTRA")

print("\n" + "="*70)
print("🎯 NGUYÊN NHÂN CHÍNH KẾT QUẢ KHÁC:")
print("="*70)
print("""
1. ⭐ Shared layer output SIZE khác (100 vs 200)
   → Network capacity khác nhau
   → Representation power khác nhau

2. 🎲 Random initialization khác
   → Có thêm T head & epsilon → seed khác
   → Starting point khác → trajectory khác

3. 📐 Model complexity khác
   → DragonNet có nhiều params hơn
   → Regularization effect khác
   → Optimization landscape khác

4. ⚠️ Dù alpha=0, beta=0:
   → Architecture vẫn khác nhau cơ bản
   → KHÔNG PHẢI do gradient flow qua propensity head
   → MÀ DO kiến trúc và capacity khác nhau!
""")

print("="*70)
print("✅ KẾT LUẬN:")
print("="*70)
print("""
Để 2 models cho kết quả GIỐNG NHAU, cần:

1. Đồng nhất shared layer output size (cùng 200 hoặc cùng 100)
2. Dùng cùng random seed
3. HOÀN TOÀN remove T head & epsilon trong Dragonnet
   (không chỉ set alpha=0, beta=0)
4. Đảm bảo architecture HOÀN TOÀN GIỐNG NHAU
""")
