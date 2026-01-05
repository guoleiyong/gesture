import os
import torch
from pathlib import Path

class Config:
    """实验配置常量 - 提分优化版"""
    # 基础路径
    BASE_DIR = Path(r"D:\ASL Alphabet")

    # 特征文件配置
    FEATURES_DIR = BASE_DIR / "reduced_features"
    SPLIT_PATH = FEATURES_DIR / "proper_dataset_split_ordered.npz"

    # --- 核心训练参数修改 ---
    BATCH_SIZE = 32      # 稳定性
    SEQ_LENGTH = 10
    INPUT_DIM = 1592
    NUM_WORKERS = 0

    # 模型配置
    EMBED_DIM = 256
    NUM_HEADS = 8
    DEPTH = 6
    DROPOUT = 0.1
    NUM_CLASSES = 29

    # 注意力机制配置
    ATTENTION_CONFIG = "alternating"  # alternating, hybrid, standard
    ATTENTION_TYPES = ["group", "linear", "efficient"]
    GROUP_SIZE = 5

    # --- 优化器配置修改 ---
    LEARNING_RATE = 5e-4  # 调大初始学习率，配合余弦退火
    WEIGHT_DECAY = 1e-4   # 防止过拟合
    WARMUP_EPOCHS = 5
    MAX_EPOCHS = 50       # 充分收敛
    PATIENCE = 25         # 早停耐心值增加

    # 优化器参数
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    GRAD_CLIP = 1.0

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULT_DIR = "results"

    @classmethod
    def get_features_dir(cls):
        return cls.FEATURES_DIR

    @classmethod
    def get_split_path(cls):
        return cls.SPLIT_PATH

def create_directories():
    """创建必要的目录"""
    directories = [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.RESULT_DIR]
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"创建目录: {dir_name}")