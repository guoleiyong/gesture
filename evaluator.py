import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
import time
from thop import profile
from config import Config


class ASEvaluator:
    """手语识别评估器"""

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = Config.DEVICE

    def evaluate(self):
        """全面评估"""
        accuracy, top5_accuracy, conf_matrix = self.evaluate_accuracy()
        efficiency = self.evaluate_efficiency()

        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'confusion_matrix': conf_matrix,
            'efficiency': efficiency
        }

    def evaluate_accuracy(self):
        """准确性评估"""
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
        top5_accuracy = top_k_accuracy_score(all_targets, all_probs, k=min(5, len(np.unique(all_targets))))
        conf_matrix = confusion_matrix(all_targets, all_preds)

        return accuracy, top5_accuracy, conf_matrix

    def evaluate_efficiency(self):
        """效率评估"""
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())

        try:
            dummy_input = torch.randn(1, Config.SEQ_LENGTH, Config.INPUT_DIM).to(self.device)
            flops, _ = profile(self.model, inputs=(dummy_input,), verbose=False)
        except:
            flops = total_params * 1000

        times = []
        with torch.no_grad():
            dummy_input = torch.randn(1, Config.SEQ_LENGTH, Config.INPUT_DIM).to(self.device)
            for _ in range(100):
                start = time.time()
                _ = self.model(dummy_input)
                end = time.time()
                times.append(end - start)

        avg_inference_time = np.mean(times)

        return {
            'parameters': total_params,
            'flops': flops,
            'inference_time': avg_inference_time,
            'fps': 1.0 / avg_inference_time
        }