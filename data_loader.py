import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from config import Config


class StaticASLSequenceDataset(Dataset):
    """静态手语时序数据集 - 增强版"""

    def __init__(self, split_name, features_dir, split_data, data_type="ideal"):
        self.features_dir = features_dir
        self.split_data = split_data
        self.data_type = data_type
        self.split_name = split_name
        self.training = (split_name == 'train')

        self.file_names = split_data[split_name]
        self.labels = split_data[f"{split_name}_labels"]

        # 构建标签映射
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        print(f"{split_name}集: {len(self.file_names)} 样本, {self.num_classes} 类别")

    def __len__(self):
        return len(self.file_names)

    def augment_features(self, features):
        """特征数据增强 - 提分关键点"""
        if not self.training:
            return features

        augmented = features.clone()

        # 1. 随机噪声 (增强强度)
        if random.random() < 0.5:
            # 噪声强度从 0.05 提升到 0.08
            noise_std = random.uniform(0.02, 0.08)
            noise = torch.randn_like(augmented) * noise_std
            augmented = augmented + noise

        # 2. 随机缩放 (增强范围)
        if random.random() < 0.5:
            # 缩放范围从 0.8-1.2 扩大到 0.7-1.3
            scale = random.uniform(0.7, 1.3)
            augmented = augmented * scale

        # 3. 随机掩码 (保持不变)
        if random.random() < 0.2:
            mask_ratio = random.uniform(0.05, 0.15)
            mask = torch.rand_like(augmented) > mask_ratio
            augmented = augmented * mask.float()

        # 4. 新增：时间偏移 (Time Shift)
        # 模拟动作开始的早晚，防止模型死记硬背第几帧是什么
        if random.random() < 0.3:
            shift = random.randint(-1, 1)  # 向前或向后移1帧
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=0)
                # 填充移动后留下的空缺
                if shift > 0:
                    augmented[:shift, :] = 0
                else:
                    augmented[shift:, :] = 0

        return augmented

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label_str = self.labels[idx]

        # 路径处理
        if '/' in file_name or '\\' in file_name:
            file_name = file_name.replace('\\', '/')
            if 'reduced_features/' in file_name:
                file_name = file_name.split('reduced_features/')[-1]
            else:
                file_name = file_name.split('/')[-1]

        if not file_name.endswith('_reduced_features.npy'):
            if file_name.endswith('.npy'):
                file_name = file_name.replace('.npy', '_reduced_features.npy')
            else:
                file_name = file_name + "_reduced_features.npy"

        feature_path = os.path.join(self.features_dir, file_name)

        try:
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()

            # 序列长度统一
            if features.shape[0] != Config.SEQ_LENGTH:
                if features.shape[0] < Config.SEQ_LENGTH:
                    padding = torch.zeros(Config.SEQ_LENGTH - features.shape[0], features.shape[1])
                    features = torch.cat([features, padding], dim=0)
                else:
                    features = features[:Config.SEQ_LENGTH]

            # 应用增强
            if self.training:
                features = self.augment_features(features)

            label_idx = self.label_to_idx[label_str]
            return features, label_idx

        except Exception as e:
            # print(f"加载失败: {feature_path}") # 减少刷屏
            dummy_features = torch.randn(Config.SEQ_LENGTH, Config.INPUT_DIM).float() * 0.1
            return dummy_features, 0


def create_data_loaders(features_dir=None, split_path=None, batch_size=None, data_type="ideal"):
    """创建数据加载器"""
    if features_dir is None:
        features_dir = Config.get_features_dir()
    if split_path is None:
        split_path = Config.get_split_path()
    batch_size = batch_size or Config.BATCH_SIZE

    try:
        split_data = np.load(split_path, allow_pickle=True)

        train_dataset = StaticASLSequenceDataset('train', features_dir, split_data, data_type)
        val_dataset = StaticASLSequenceDataset('val', features_dir, split_data, data_type)
        test_dataset = StaticASLSequenceDataset('test', features_dir, split_data, data_type)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS)

        return train_loader, val_loader, test_loader, train_dataset

    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None