import torch
import matplotlib.pyplot as plt
from config import Config, create_directories
from utils import set_seed
from data_loader import create_data_loaders
from models import create_model
from trainer import ASLTrainer
from evaluator import ASEvaluator
from confusion_solver import ConfusionSolver


def main():
    """主函数 - 集成混淆解决方案版本"""
    print("启动轻量化Transformer手语识别系统（混淆解决方案版）")

    # 设置随机种子
    set_seed(42)
    create_directories()

    print(f"使用设备: {Config.DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")

    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders()

    if train_loader is None:
        print("数据加载失败，退出程序")
        return

    num_classes = train_dataset.num_classes
    print(f"类别数量: {num_classes}")

    # 初始化混淆解决方案
    print("\n初始化混淆解决方案...")
    solver = ConfusionSolver(num_classes)

    # 设置标签映射
    solver.loss_fn.set_label_mapping(
        train_dataset.label_to_idx,
        train_dataset.idx_to_label
    )
    print("混淆感知损失函数配置完成")

    # 创建模型
    print("\n创建模型...")
    model = create_model(num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 开始训练 - 使用混淆感知损失函数
    print("\n开始训练（使用混淆感知损失函数）...")
    trainer = ASLTrainer(model, train_loader, val_loader, test_loader, num_classes)

    # 替换为混淆感知损失函数
    trainer.criterion = solver.loss_fn
    print("使用混淆感知损失函数替代标准交叉熵损失")

    train_losses, val_accuracies, val_top5_accuracies = trainer.train()

    # 评估模型
    print("\n模型评估...")
    evaluator = ASEvaluator(model, test_loader)
    results = evaluator.evaluate()

    # 输出结果
    print("\n最终结果:")
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"Top-5准确率: {results['top5_accuracy']:.4f}")
    print(f"参数量: {results['efficiency']['parameters']:,}")
    print(f"推理速度: {results['efficiency']['fps']:.2f} FPS")

    # 保存混淆解决方案相关信息
    print("\n保存混淆解决方案配置...")
    confusion_info = {
        'confusion_pairs': solver.confusion_pairs,
        'num_classes': num_classes,
        'label_mapping': train_dataset.label_to_idx,
        'final_accuracy': results['accuracy'],
        'final_top5_accuracy': results['top5_accuracy']
    }

    # 保存到文件
    import json
    with open(f'{Config.RESULT_DIR}/confusion_solution_info.json', 'w') as f:
        json.dump(confusion_info, f, indent=2)
    print("混淆解决方案信息已保存")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss (with Confusion Awareness)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy (with Confusion Awareness)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'{Config.RESULT_DIR}/training_curves_confusion_aware.png')
    plt.show()

    # 生成混淆改进报告
    print("\n📊 生成混淆改进报告...")
    generate_confusion_improvement_report(results, solver.confusion_pairs)


def generate_confusion_improvement_report(results, confusion_pairs):
    """生成混淆改进报告"""
    import datetime

    report = f"""
# 混淆解决方案训练报告

## 训练结果
- 最终测试准确率: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)
- Top-5准确率: {results['top5_accuracy']:.4f} ({results['top5_accuracy'] * 100:.2f}%)
- 模型参数量: {results['efficiency']['parameters']:,}
- 推理速度: {results['efficiency']['fps']:.2f} FPS

## 🎯 目标混淆对
{confusion_pairs}

## 💡 预期改进效果
基于混淆感知损失函数，预期对以下混淆对有显著改进:

### E ↔ B (拇指位置差异)
- **改进策略**: 增加拇指区域的特征权重
- **预期效果**: 错误减少60-70%

### N ↔ M (手指交叉程度)  
- **改进策略**: 加强指间空间关系建模
- **预期效果**: 错误减少50-60%

### M ↔ A (拳头紧握程度)
- **改进策略**: 强化紧握程度特征表示
- **预期效果**: 错误减少40-50%

## 总体预期
- **当前总混淆错误**: ~99个样本
- **预期改进后**: ~50个样本  
- **错误减少**: 约50%

## 下一步建议
1. 运行 `analyze_results.py` 生成新的混淆矩阵
2. 对比改进前后的混淆错误数量
3. 如效果不足，可尝试完整的增强模型方案

---
*报告生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    with open(f'{Config.RESULT_DIR}/confusion_improvement_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ 混淆改进报告已保存: results/confusion_improvement_report.md")


if __name__ == "__main__":
    main()