import torch
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from config import Config


def set_seed(seed=42):
    """设置随机种子确保实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adapt_feature_dimension(features, target_dim=1592):
    """将特征适配到目标维度"""
    current_dim = features.shape[1]

    if current_dim == target_dim:
        return features

    if current_dim < target_dim:
        repeat_times = target_dim // current_dim
        remainder = target_dim % current_dim

        adapted_features = np.tile(features, (1, repeat_times))
        if remainder > 0:
            padding = features[:, :remainder]
            adapted_features = np.concatenate([adapted_features, padding], axis=1)
        return adapted_features
    else:
        return features[:, :target_dim]


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def create_proper_split():
    """创建正确的数据划分（60%训练，20%验证，20%测试）- 保持文件顺序版本"""
    print("创建正确的数据划分 (保持文件顺序)...")

    features_dir = Config.IDEAL_FEATURES_DIR
    files = [f for f in os.listdir(features_dir) if f.endswith('_reduced_features.npy')]

    # 按文件名排序，确保顺序一致
    files.sort()
    print(f"找到 {len(files)} 个特征文件")
    print(f"前5个文件: {files[:5]}")
    print(f"后5个文件: {files[-5:]}")

    # 提取标签
    labels = [f.split('_')[0] for f in files]

    # 检查标签分布
    from collections import Counter
    label_count = Counter(labels)
    print(f"标签分布: {len(label_count)} 个类别")
    for label, count in list(label_count.items())[:10]:
        print(f"  {label}: {count} 样本")

    # 按类别分层划分 - 保持文件顺序
    unique_labels = sorted(label_count.keys())  # 按字母排序

    train_files = []
    val_files = []
    test_files = []
    train_labels = []
    val_labels = []
    test_labels = []

    for label in unique_labels:
        # 获取该类别的所有文件（已经是排序的）
        label_files = [f for f in files if f.startswith(label + '_')]

        # 计算划分点（保持顺序，不随机打乱）
        total = len(label_files)
        train_end = int(total * 0.6)  # 前60%训练
        val_end = train_end + int(total * 0.2)  # 接下来20%验证

        train = label_files[:train_end]
        val = label_files[train_end:val_end]
        test = label_files[val_end:]

        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

        train_labels.extend([label] * len(train))
        val_labels.extend([label] * len(val))
        test_labels.extend([label] * len(test))

    # 验证划分正确性
    print(f"\n顺序保持划分验证:")
    print(f"训练集: {len(train_files)} 文件")
    print(f"验证集: {len(val_files)} 文件")
    print(f"测试集: {len(test_files)} 文件")

    # 检查每个集合的文件顺序
    print(f"\n训练集前5个文件: {train_files[:5]}")
    print(f"验证集前5个文件: {val_files[:5]}")
    print(f"测试集前5个文件: {test_files[:5]}")

    # 检查每个集合的标签分布
    print(f"\n训练集标签分布: {Counter(train_labels)}")
    print(f"验证集标签分布: {Counter(val_labels)}")
    print(f"测试集标签分布: {Counter(test_labels)}")

    # 保存新划分
    proper_split = {
        'train': np.array(train_files),
        'train_labels': np.array(train_labels),
        'val': np.array(val_files),
        'val_labels': np.array(val_labels),
        'test': np.array(test_files),
        'test_labels': np.array(test_labels)
    }

    split_path = Config.BASE_DIR / "proper_dataset_split_ordered.npz"  # 新文件名
    np.savez(split_path, **proper_split)

    print(f"\n正确划分完成!")
    print(f"训练集: {len(train_files)} 样本")
    print(f"验证集: {len(val_files)} 样本")
    print(f"测试集: {len(test_files)} 样本")
    print(f"保存到: {split_path}")

    # 验证划分比例
    total = len(train_files) + len(val_files) + len(test_files)
    print(f"\n划分比例:")
    print(f"训练集: {len(train_files) / total:.1%} ({len(train_files)}/{total})")
    print(f"验证集: {len(val_files) / total:.1%} ({len(val_files)}/{total})")
    print(f"测试集: {len(test_files) / total:.1%} ({len(test_files)}/{total})")

# 运行这个函数来创建新划分
if __name__ == "__main__":
    create_proper_split()