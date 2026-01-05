import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from config import Config
from models import create_model
from data_loader import create_data_loaders


def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体设置成功")
        return True
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用英文字体替代")
            return False
        except:
            print("字体设置失败")
            return False


# 设置中文字体
chinese_supported = setup_chinese_font()

COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']


def get_temporal_importance(model, features, target_class):
    """计算时间重要性分数"""
    model.eval()
    features = features.clone().detach()
    features.requires_grad = True
    output = model(features)
    score = output[0, target_class]
    score.backward()
    if features.grad is None:
        return np.zeros(10)
    gradients = features.grad.data.abs()[0]
    temporal_importance = gradients.mean(dim=1).cpu().numpy()
    if temporal_importance.max() > 0:
        temporal_importance = (temporal_importance - temporal_importance.min()) / (
                temporal_importance.max() - temporal_importance.min() + 1e-8)
    return temporal_importance


def visualize_samples():
    print(" 开始生成可解释性分析图...")
    device = Config.DEVICE

    # 1. 加载资源
    print(" 加载数据与模型...")
    _, _, test_loader, train_dataset = create_data_loaders()
    model = create_model(train_dataset.num_classes).to(device)
    checkpoint = torch.load(f'{Config.CHECKPOINT_DIR}/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. 强制寻找4个样本 (修复版逻辑)
    # 我们希望尽量找这几个，找不到就随机替补
    preferred_labels = ['J', 'Z', 'A', 'M']
    collected_samples = []  # 列表结构: [(name, feature, label_idx)]
    collected_names = set()

    print(f" 正在搜索样本，目标: {preferred_labels}")

    # 遍历整个测试集 (为了确保找到，我们多跑一点)
    for batch_idx, (features, labels) in enumerate(test_loader):
        if len(collected_samples) >= 4:
            break

        for i in range(len(labels)):
            if len(collected_samples) >= 4:
                break

            idx = labels[i].item()
            name = train_dataset.idx_to_label[idx]

            # 逻辑：如果是目标类别，且还没收集过，就收集
            if name in preferred_labels and name not in collected_names:
                collected_samples.append((name, features[i].unsqueeze(0).to(device), idx))
                collected_names.add(name)
                print(f" 找到目标样本: {name}")

    # 如果还没满4个，就从测试集里随便抓几个不重复的补满
    if len(collected_samples) < 4:
        print(" 目标样本未找齐，正在补充随机样本...")
        data_iter = iter(test_loader)
        while len(collected_samples) < 4:
            try:
                features, labels = next(data_iter)
                for i in range(len(labels)):
                    if len(collected_samples) >= 4:
                        break
                    idx = labels[i].item()
                    name = train_dataset.idx_to_label[idx]

                    # 只要名字不重复就加进去
                    if name not in collected_names:
                        collected_samples.append((name, features[i].unsqueeze(0).to(device), idx))
                        collected_names.add(name)
                        print(f"   ➕ 补充样本: {name}")
            except StopIteration:
                break

    # 3. 绘图
    print(" 正在绘图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    for idx, (name, feat, label_idx) in enumerate(collected_samples):
        ax = axes[idx]
        importance = get_temporal_importance(model, feat, label_idx)

        frames = np.arange(1, 11)
        bars = ax.bar(frames, importance, color=COLORS[idx % len(COLORS)], alpha=0.8, width=0.7)
        ax.plot(frames, importance, color='gray', linestyle='--', alpha=0.5)

        # 设置中文或英文标题
        if chinese_supported:
            ax.set_title(f'手势类别: "{name}" - 时间注意力分析', fontsize=14, fontweight='bold')
            ax.set_xlabel('时间步 (帧数)', fontsize=10)
            ax.set_ylabel('重要性分数', fontsize=10)
        else:
            ax.set_title(f'Class "{name}" - Temporal Attention Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step (Frame)', fontsize=10)
            ax.set_ylabel('Importance Score', fontsize=10)

        ax.set_ylim(0, 1.15)
        ax.set_xticks(frames)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 标注关键帧
        max_idx = np.argmax(importance)
        bars[max_idx].set_color('#D62728')
        bars[max_idx].set_alpha(1.0)

        if chinese_supported:
            ax.text(frames[max_idx], importance[max_idx] + 0.02, '关键帧',
                    ha='center', fontsize=9, color='#D62728', fontweight='bold')
        else:
            ax.text(frames[max_idx], importance[max_idx] + 0.02, 'Key Frame',
                    ha='center', fontsize=9, color='#D62728', fontweight='bold')

        # 添加数值标签
        for j, (frame, value) in enumerate(zip(frames, importance)):
            if value > 0.1:  # 只显示较大的值
                ax.text(frame, value + 0.02, f'{value:.2f}',
                        ha='center', fontsize=8, color='black')

    # 如果实在凑不齐4个（极小概率），把多余的轴关掉
    for i in range(len(collected_samples), 4):
        axes[i].axis('off')

    # 添加整体标题
    if chinese_supported:
        fig.suptitle('混合注意力模型 - 时间维度可解释性分析', fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Hybrid Attention Model - Temporal Interpretability Analysis',
                     fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存图片
    if chinese_supported:
        output_path = 'results/时间注意力分析_中文.png'
    else:
        output_path = 'results/temporal_importance_analysis.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" 完成！图表已保存: {output_path}")
    plt.show()

    # 生成分析报告
    generate_analysis_report(collected_samples, model)


def generate_analysis_report(samples, model):
    """生成详细的分析报告"""
    print("\n 时间注意力分析报告:")
    print("=" * 60)

    for name, feat, label_idx in samples:
        importance = get_temporal_importance(model, feat, label_idx)
        max_idx = np.argmax(importance)
        avg_importance = np.mean(importance)

        if chinese_supported:
            print(f"手势: {name}")
            print(f"  关键帧位置: 第{max_idx + 1}帧 (重要性: {importance[max_idx]:.3f})")
            print(f"  平均重要性: {avg_importance:.3f}")
            print(f"  重要性分布: {', '.join([f'{i + 1}:{v:.2f}' for i, v in enumerate(importance) if v > 0.2])}")
        else:
            print(f"Gesture: {name}")
            print(f"  Key Frame: Frame {max_idx + 1} (Importance: {importance[max_idx]:.3f})")
            print(f"  Average Importance: {avg_importance:.3f}")
            print(f"  Distribution: {', '.join([f'{i + 1}:{v:.2f}' for i, v in enumerate(importance) if v > 0.2])}")
        print()


def visualize_comparative_analysis():
    """生成对比分析图：不同手势的时间注意力对比"""
    print("\n 生成对比分析图...")

    device = Config.DEVICE

    # 加载数据与模型
    _, _, test_loader, train_dataset = create_data_loaders()
    model = create_model(train_dataset.num_classes).to(device)
    checkpoint = torch.load(f'{Config.CHECKPOINT_DIR}/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 选择4个有代表性的手势
    target_labels = ['A', 'B', 'C', 'D']
    samples = []

    # 寻找样本
    for features, labels in test_loader:
        if len(samples) >= 4:
            break
        for i in range(len(labels)):
            if len(samples) >= 4:
                break
            idx = labels[i].item()
            name = train_dataset.idx_to_label[idx]
            if name in target_labels and name not in [s[0] for s in samples]:
                samples.append((name, features[i].unsqueeze(0).to(device), idx))

    # 创建对比图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for idx, (name, feat, label_idx) in enumerate(samples):
        importance = get_temporal_importance(model, feat, label_idx)
        frames = np.arange(1, 11)

        # 使用不同的线型和标记
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']

        ax.plot(frames, importance, line_styles[idx], marker=markers[idx],
                label=f'{name}', linewidth=2, markersize=8, color=COLORS[idx])

    # 设置标签
    if chinese_supported:
        ax.set_title('不同手势的时间注意力对比分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步 (帧数)', fontsize=12)
        ax.set_ylabel('重要性分数', fontsize=12)
    else:
        ax.set_title('Comparative Analysis of Temporal Attention', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step (Frame)', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)

    ax.set_xticks(frames)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # 保存
    if chinese_supported:
        output_path = 'results/时间注意力对比分析.png'
    else:
        output_path = 'results/temporal_comparison_analysis.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" 对比分析图已保存: {output_path}")
    plt.show()


if __name__ == "__main__":
    print(" 注意力可视化分析")
    print("=" * 60)

    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)

    # 生成主要图表
    visualize_samples()

    # 生成对比分析图
    visualize_comparative_analysis()

    print("\n 所有可视化图表生成完成!")
    print(" 生成的文件:")
    print("   - results/时间注意力分析_中文.png (主要分析)")
    print("   - results/时间注意力对比分析.png (对比分析)")

    if not chinese_supported:
        print("\n  注意: 中文字体不可用，图表使用英文标签")
        print(" 建议安装中文字体以获得更好的显示效果")