# plot_confusion_matrix.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 导入项目中的模块
from config import Config
from models import create_model
from data_loader import create_data_loaders


def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体设置成功")
        return True
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠使用英文字体替代")
            return False
        except:
            print("字体设置失败")
            return False


def plot_cm():
    print("开始生成混淆矩阵...")

    # 设置中文字体
    chinese_supported = setup_chinese_font()

    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    # 2. 加载数据 (只用测试集)
    print("   加载数据...")
    _, _, test_loader, train_dataset = create_data_loaders()

    # 获取类别名称 (例如 ['A', 'B', 'C', ...])
    # 如果 dataset 中有 idx_to_label，我们就用它，否则用数字
    if hasattr(train_dataset, 'idx_to_label'):
        class_names = [train_dataset.idx_to_label[i] for i in range(train_dataset.num_classes)]
    else:
        class_names = [str(i) for i in range(train_dataset.num_classes)]

    print(f"   类别列表: {class_names}")

    # 3. 加载模型
    print("   加载最佳模型权重...")
    model = create_model(train_dataset.num_classes).to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   成功加载模型 (验证准确率: {checkpoint.get('best_accuracy', 0):.4f})")
    else:
        print(f"   未找到模型文件: {checkpoint_path}")
        return

    # 4. 进行预测
    model.eval()
    all_preds = []
    all_targets = []

    print("   正在进行推理...")
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # 5. 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    # 6. 绘图 (调整样式以匹配你提供的图片，但更美观)
    plt.figure(figsize=(20, 18))  # 画布大一点，因为有29个类

    # 使用 Seaborn 画热力图
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names,
                     yticklabels=class_names,
                     annot_kws={"size": 10},  # 数字字体大小
                     cbar_kws={'label': '样本数量'})  # 颜色条标签

    # 设置中文标题和标签
    if chinese_supported:
        plt.title('混淆矩阵 - ASL手语字母识别', fontsize=20, pad=20, fontweight='bold')
        plt.ylabel('真实标签', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=16, fontweight='bold')
    else:
        plt.title('Confusion Matrix - ASL Alphabet Recognition', fontsize=20, pad=20, fontweight='bold')
        plt.ylabel('True Label', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')

    # 调整刻度字体
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 保存高清图
    save_path = os.path.join(Config.RESULT_DIR, 'final_confusion_matrix_chinese.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至: {save_path}")

    plt.show()

    # 7. 额外分析：计算各类别准确率
    print("\n各类别准确率分析:")
    print("=" * 60)

    # 计算各类别的准确率（对角线值除以该行总和）
    class_accuracies = []
    for i in range(len(class_names)):
        total = np.sum(cm[i, :])  # 该类别总样本数
        correct = cm[i, i]  # 正确预测数
        accuracy = correct / total if total > 0 else 0
        class_accuracies.append(accuracy)

        # 打印每个类别的准确率
        print(f"{class_names[i]:<5}: {accuracy:.4f} ({accuracy * 100:5.2f}%) - 样本数: {total}")

    # 总体准确率
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n总体准确率: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # 找出最容易混淆的类别对
    print("\n最常混淆的类别对 (前5名):")
    print("-" * 60)
    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_pairs.append((class_names[i], class_names[j], cm[i, j]))

    # 按错误数量排序
    error_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (true_class, pred_class, count) in enumerate(error_pairs[:5], 1):
        print(f"{i}. {true_class} → {pred_class}: {count}次")

    # 生成简化的主要错误混淆矩阵
    if chinese_supported:
        create_simplified_confusion_matrix(cm, class_names, chinese_supported)

    return cm, class_names, overall_accuracy


def create_simplified_confusion_matrix(cm, class_names, chinese_supported):
    """创建简化的主要错误混淆矩阵"""
    plt.figure(figsize=(16, 14))

    # 创建一个掩码，只显示错误数大于阈值的格子
    threshold = 3  # 只显示错误数>=3的格子
    mask = cm < threshold
    np.fill_diagonal(mask, False)  # 总是显示对角线（正确预测）

    # 使用红色调色板突出显示错误
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names,
                yticklabels=class_names,
                mask=mask,  # 应用掩码
                cbar_kws={'label': '样本数量'})

    if chinese_supported:
        plt.title('混淆矩阵 - 主要错误分析 (≥3次误分类)',
                  fontsize=18, pad=20, fontweight='bold')
        plt.ylabel('真实标签', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=14, fontweight='bold')
    else:
        plt.title('Confusion Matrix - Major Errors (≥3 misclassifications)',
                  fontsize=18, pad=20, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # 保存简化的混淆矩阵
    simplified_path = os.path.join(Config.RESULT_DIR, 'confusion_matrix_major_errors_chinese.png')
    plt.savefig(simplified_path, dpi=300, bbox_inches='tight')
    print(f"主要错误混淆矩阵已保存: {simplified_path}")

    plt.show()


def save_analysis_report(cm, class_names, overall_accuracy):
    """保存详细的分析报告"""
    report_path = os.path.join(Config.RESULT_DIR, 'confusion_matrix_analysis.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("混淆矩阵分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)\n\n")

        f.write("各类别准确率:\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            total = np.sum(cm[i, :])
            correct = cm[i, i]
            accuracy = correct / total if total > 0 else 0
            f.write(f"{class_name:<5}: {accuracy:.4f} ({accuracy * 100:5.2f}%) - 样本数: {total}\n")

        f.write("\n最常混淆的类别对:\n")
        f.write("-" * 60 + "\n")
        error_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    error_pairs.append((class_names[i], class_names[j], cm[i, j]))

        error_pairs.sort(key=lambda x: x[2], reverse=True)
        for i, (true_class, pred_class, count) in enumerate(error_pairs[:10], 1):
            f.write(f"{i}. {true_class} → {pred_class}: {count}次\n")

        f.write("\n混淆矩阵统计:\n")
        f.write("-" * 60 + "\n")
        f.write(f"总样本数: {np.sum(cm)}\n")
        f.write(f"正确预测数: {np.trace(cm)}\n")
        f.write(f"错误预测数: {np.sum(cm) - np.trace(cm)}\n")
        f.write(f"错误率: {(np.sum(cm) - np.trace(cm)) / np.sum(cm):.4f}\n")

    print(f"详细分析报告已保存: {report_path}")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    print("开始生成中文混淆矩阵...")
    print("确保已安装中文字体: SimHei 或 Microsoft YaHei")
    print("-" * 60)

    # 生成混淆矩阵
    cm, class_names, accuracy = plot_cm()

    # 保存分析报告
    save_analysis_report(cm, class_names, accuracy)

    print("\n" + "=" * 60)
    print("混淆矩阵分析完成!")
    print("生成的文件:")
    print(f"   1. {os.path.join(Config.RESULT_DIR, 'final_confusion_matrix_chinese.png')}")
    print(f"   2. {os.path.join(Config.RESULT_DIR, 'confusion_matrix_major_errors_chinese.png')}")
    print(f"   3. {os.path.join(Config.RESULT_DIR, 'confusion_matrix_analysis.txt')}")
    print(f"\n最终结果: 总体准确率 {accuracy:.4f} ({accuracy * 100:.2f}%)")