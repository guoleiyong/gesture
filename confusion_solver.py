# confusion_solver.py
"""
针对手语混淆问题的专项解决方案 - 核心组件
重点解决: E↔B, N↔M, M↔A 等易混淆对
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionAwareLoss(nn.Module):
    """混淆感知的损失函数 - 为重点混淆对增加惩罚"""

    def __init__(self, num_classes, confusion_pairs, alpha=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_pairs = confusion_pairs
        self.alpha = alpha
        self.base_loss = nn.CrossEntropyLoss()
        self.label_to_idx = {}

    def set_label_mapping(self, label_to_idx, idx_to_label):
        """设置标签映射"""
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label

    def forward(self, logits, targets):
        base_loss = self.base_loss(logits, targets)

        if not self.label_to_idx:
            return base_loss

        confusion_loss = 0.0
        pair_count = 0

        for class1, class2 in self.confusion_pairs:
            if class1 in self.label_to_idx and class2 in self.label_to_idx:
                idx1 = self.label_to_idx[class1]
                idx2 = self.label_to_idx[class2]

                # 找出当前batch中属于这两个类别的样本
                mask = (targets == idx1) | (targets == idx2)
                if mask.sum() > 0:
                    confusion_logits = logits[mask]
                    confusion_targets = targets[mask]

                    # 对这些样本单独计算损失
                    pair_loss = F.cross_entropy(confusion_logits, confusion_targets)
                    confusion_loss += pair_loss
                    pair_count += 1

        if pair_count > 0:
            confusion_loss = confusion_loss / pair_count
            total_loss = base_loss + self.alpha * confusion_loss
        else:
            total_loss = base_loss

        return total_loss


class FineGrainedAttention(nn.Module):
    """细粒度注意力 - 针对易混淆区域的专项注意力"""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.confusion_queries = nn.Parameter(torch.randn(3, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        # 简化计算逻辑，只保留核心特征提取
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 针对3个混淆对的特征提取
        confusion_features = []
        for i in range(3):
            conf_q = self.confusion_queries[i].view(1, 1, self.embed_dim).expand(B, -1, -1)
            conf_q = conf_q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

            attn = (conf_q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            feat = (attn @ v).transpose(1, 2).reshape(B, 1, self.embed_dim)
            confusion_features.append(feat)

        confusion_feat = torch.cat(confusion_features, dim=1).mean(dim=1)  # [B, embed_dim]

        # 同时也计算正常的自注意力
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        return out, confusion_feat


class ConfusionAwareModel(nn.Module):
    """把Attention整合进主模型 - 暂未在main中启用，留作扩展"""
    pass  # 暂时留空，main.py目前使用的是models.py里的模型


class TargetedDataAugmentation:
    """针对易混淆类别的数据增强"""

    def __init__(self, confusion_pairs):
        self.confusion_pairs = confusion_pairs

    def augment_for_pair(self, features, label):
        """为特定混淆对生成增强样本"""
        aug_features = features.clone()

        # 针对特定类别添加特定噪声
        if label in ['E', 'B']:  # 拇指变化
            noise = torch.randn_like(features) * 0.05
            features[:, :, 100:200] += noise[:, :, 100:200] * 2
        elif label in ['N', 'M']:  # 手指交叉
            scale = torch.rand(1) * 0.3 + 0.85
            features[:, :, 300:500] *= scale

        return aug_features


class ConfusionSolver:
    """混淆问题解决器 - 主控制器"""

    def __init__(self, num_classes=29):
        self.num_classes = num_classes
        # 定义最容易混淆的类别对
        self.confusion_pairs = [('E', 'B'), ('N', 'M'), ('M', 'A')]

        self.loss_fn = ConfusionAwareLoss(num_classes, self.confusion_pairs)
        self.data_augmenter = TargetedDataAugmentation(self.confusion_pairs)

        print(f"混淆解决器就绪: 针对 {self.confusion_pairs}")

    def augment_batch(self, features, labels, label_mapping=None):
        """对batch进行针对性增强 - 暂未在main中启用，留作扩展"""
        return features, labels