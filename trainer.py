import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from config import Config


class ASLTrainer:
    """手语识别训练器 - 策略优化版"""

    def __init__(self, model, train_loader, val_loader, test_loader=None, num_classes=29):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = Config.DEVICE
        self.num_classes = num_classes

        # 默认损失函数 (会被 main.py 中的 ConfusionAwareLoss 替换)
        self.criterion = nn.CrossEntropyLoss()

        # 1. 优化器: AdamW (权重衰减修复)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            eps=1e-8
        )

        # 2. 学习率调度器: 余弦退火 (Cosine Annealing) <--- 核心修改
        # 这会让学习率像波浪一样下降，有助于找到更好的全局最优解
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 第一个周期为10个epoch
            T_mult=2,  # 之后周期倍增 (10 -> 20 -> 40...)
            eta_min=1e-6  # 最小学习率
        )

        self.train_losses = []
        self.val_accuracies = []
        self.val_top5_accuracies = []
        self.learning_rates = []

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP)
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

        epoch_loss = running_loss / total_samples
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        return epoch_loss

    def validate(self):
        """验证"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)

                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)

        if self.num_classes >= 5:
            try:
                top5_accuracy = top_k_accuracy_score(all_targets, all_probs, k=5)
            except:
                top5_accuracy = accuracy
        else:
            top5_accuracy = accuracy

        return accuracy, top5_accuracy

    def test(self):
        """测试模型"""
        if self.test_loader is None:
            return 0.0, 0.0

        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        if self.num_classes >= 5:
            try:
                top5_accuracy = top_k_accuracy_score(all_targets, all_probs, k=5)
            except:
                top5_accuracy = accuracy
        else:
            top5_accuracy = accuracy

        return accuracy, top5_accuracy

    def train(self):
        """完整训练流程"""
        print(f"开始训练，共{Config.MAX_EPOCHS}个epoch，策略: 余弦退火")

        best_accuracy = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(Config.MAX_EPOCHS):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_accuracy, val_top5_accuracy = self.validate()
            self.val_accuracies.append(val_accuracy)
            self.val_top5_accuracies.append(val_top5_accuracy)

            # 更新学习率 (CosineAnnealing 需要在每个epoch结束调用)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start

            print(f'Epoch {epoch + 1:02d} | '
                  f'Loss: {train_loss:.4f} | '
                  f'Acc: {val_accuracy:.4f} | '
                  f'Top5: {val_top5_accuracy:.4f} | '
                  f'LR: {self.learning_rates[-1]:.2e} | '
                  f'Time: {epoch_time:.1f}s')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'best_accuracy': best_accuracy,
                }, f'{Config.CHECKPOINT_DIR}/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f'\n早停触发 (Patience={Config.PATIENCE})')
                    break

        print(f"\n最佳验证准确率: {best_accuracy:.4f}")

        # 最终测试
        if self.test_loader is not None:
            print("\n正在进行最终测试...")
            test_acc, test_top5 = self.test()
            print(f'最终测试集准确率: {test_acc:.4f}')
            print(f'最终测试集Top-5: {test_top5:.4f}')

        return self.train_losses, self.val_accuracies, self.val_top5_accuracies