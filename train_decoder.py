import os
import torch
import cloudy

def main():
    # 可根据需要覆盖默认配置
    pipeline = cloudy.create_pipeline(
        workspace='./workspace_default',   # 训练产生的 ckpt 会保存在这里
        data_path='/data/cloudy',  # 改成你的数据路径
        clouds_folder='cloudy',
    )

    # 直接开训（内部已处理 DDP、多卡划分与断点续训）
    pipeline.run_train_decoder()

if __name__ == '__main__':
    # 便于在 Windows/WSL 混用
    torch.backends.cudnn.benchmark = True
    main()
