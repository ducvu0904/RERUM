import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

def get_sampler(y_train, target_positive_ratio=0.5):
    """
    Tạo WeightedRandomSampler để cân bằng lại dữ liệu trong mỗi Batch.
    
    Parameters:
    ----------
    y_train : torch.Tensor hoặc numpy array
        Dữ liệu target gốc (số tiền chi tiêu).
    target_positive_ratio : float (default=0.2)
        Tỷ lệ mẫu dương (người mua) mong muốn trong mỗi batch.
        0.2 nghĩa là muốn 20% batch là người mua, 80% là không mua.
        Nên để từ 0.1 đến 0.3 để tránh model bị ảo giác (overfit).
    
    Returns:
    -------
    sampler : WeightedRandomSampler
        Sampler để đưa vào DataLoader.
    """
    
    # 1. Đảm bảo y_train là Tensor 1 chiều
    if not torch.is_tensor(y_train):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    y_train = y_train.view(-1) 
    
    # 2. Tạo nhãn nhị phân tạm thời (0: Không mua, 1: Mua)
    # Lưu ý: Giá trị tiền vẫn giữ nguyên trong dataset, đây chỉ là nhãn để tính weight
    targets = (y_train > 0).long()
    
    # 3. Đếm số lượng từng class
    count_0 = (targets == 0).sum().item()
    count_1 = (targets == 1).sum().item()
    
    print(f"📊 [Sampler Info] Gốc: Không mua = {count_0}, Có mua = {count_1}")
    
    if count_1 == 0:
        raise ValueError("Lỗi: Tập train không có người mua nào (y > 0)!")

    # 4. Tính trọng số cho từng CLASS dựa trên tỷ lệ mong muốn
    # Công thức: Weight = Tỷ lệ mong muốn / Số lượng thực tế
    weight_for_0 = (1.0 - target_positive_ratio) / count_0
    weight_for_1 = target_positive_ratio / count_1
    
    print(f"⚖️ [Sampler Info] Tỷ lệ mục tiêu: {target_positive_ratio*100}% Mua")
    print(f"   -> Weight class 0: {weight_for_0:.6f}")
    print(f"   -> Weight class 1: {weight_for_1:.6f} (Gấp {weight_for_1/weight_for_0:.1f} lần)")

    # 5. Gán trọng số cho từng MẪU (Sample Weights)
    sample_weights = torch.zeros_like(y_train, dtype=torch.float)
    sample_weights[targets == 0] = weight_for_0
    sample_weights[targets == 1] = weight_for_1
    
    # 6. Tạo Sampler
    # num_samples: Tổng số mẫu muốn bốc trong 1 epoch (thường bằng len data gốc)
    # replacement=True: BẮT BUỘC để có thể bốc lặp lại các mẫu hiếm
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler