import torch
visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
healthy_ids = []
if visible > 0:
    for i in range(visible):
        try:
            torch.cuda.set_device(i)
            _ = torch.empty(1, device=f"cuda:{i}")  # 小额分配试探
            healthy_ids.append(i)
        except Exception as e:
            print(f"[WARN] skip cuda:{i}: {e}")
    if not healthy_ids:
        healthy_ids = [0]
    # 设定默认设备为第一个健康 GPU
    torch.cuda.set_device(healthy_ids[0])
    print(f"Visible CUDA: {visible} Using device_ids: {healthy_ids}")
else:
    print("No CUDA available, running on CPU.")
    healthy_ids = []
