import torch
# 假设您的数据集放在 /my_dataset/cloudy/ 下，其中包含 cloud_0.pt, cloud_1.pt … 等体密度文件
# workspace 指向一个新的目录，用于存储训练过程的中间结果
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
import cloudy
#device = self.get_device()  # 通常为 torch.device('cuda') 或 'cpu'
pipeline = cloudy.create_pipeline(
    workspace='./workspace_default',
    data_path='/data/cloudy',      # 指向包含云体文件的根目录
    clouds_folder='cloudy',
)# 子目录名，里面应有 cloud_0.pt 等
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

#device = self.get_device()  # 通常为 torch.device('cuda') 或 'cpu'
pipeline.run_train_decoder()

# 2. 将所有云体编码成 latent，并保存为 workspace/encoded/latent_*.pt
#    如果首次运行，请显式指定 start_id=0
pipeline.run_encoding(start_id=0, end_id=pipeline.get_number_of_clouds())

# # 3. 对每个 latent 做 14 种旋转/缩放增强，生成训练扩散模型所需的更多样本
# #    必须指定 start_cloud，否则 start_cloud 为 None 时源码中的 start_cloud * 14 会抛错:contentReference[oaicite:0]{index=0}。
pipeline.run_enhancing(start_cloud=0, end_cloud=pipeline.get_number_of_clouds())

# # 4. 计算增强后所有 latent 的均值和尺度，用于标准化扩散模型输入
pipeline.run_compute_normalization_stats()

# # 5. 训练扩散模型，使其能够生成与潜在分布一致的 latent 表示
# #    训练前会检查增强后的 latent 数量不少于 train_N:contentReference[oaicite:1]{index=1}。
pipeline.run_train_diffuser()

