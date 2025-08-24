import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = 'datasets/data.yaml'  # Put your dataset config here
    check_path(yaml_path)
    # Load the trained model as the starting point for incremental training
    model = RTDETR('runs/train/exp6/weights/last.pt')  # Load trained model instead of architecture
    model.train(data=str(yaml_path),
                cache=False,
                imgsz=640,
                # Incremental training - start from trained model
                epochs=1,  # Train for 1 epoch starting from the trained model
                batch=1,
                workers=2,
                device='0',
                # resume='runs/train/exp6/weights/last.pt',  # Don't use resume, load model directly above
                project='runs/train',
                name='exp',
                patience = 40, # early stopping patience
                )