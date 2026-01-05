import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy

# 导入你的核心模块
from config import Config
from models import create_model
from trainer import ASLTrainer
from evaluator import ASEvaluator
from data_loader import create_data_loaders
from utils import set_seed

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def run_benchmark():
    print("开始运行全模型对比基准测试...")

    # 1. 准备数据 (只加载一次)
    print("加载数据...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders()
    num_classes = train_dataset.num_classes

    # 2. 定义要对比的实验配置
    # 键是代码里的配置名，值是图表上显示的名字
    experiments = {
        'group': '仅分组注意力 (Group)',
        'efficient': '仅线性注意力 (Linear)',
        'alternating': '我们的方法 (Ours)'
    }

    # 用于存储结果
    results = {
        'names': [],
        'accuracy': [],
        'params': [],
        'fps': []
    }

    # 保存原始配置以便恢复
    original_epochs = Config.MAX_EPOCHS

    TEST_EPOCHS = 50
    Config.MAX_EPOCHS = TEST_EPOCHS

    print(f"\n每个模型将训练 {TEST_EPOCHS} 个 Epoch 用于对比...")

    # 3. 循环运行实验
    for config_name, display_name in experiments.items():
        print(f"\n" + "=" * 50)
        print(f"▶正在测试模型: {display_name}")
        print("=" * 50)

        # 修改配置
        Config.ATTENTION_CONFIG = config_name

        # 创建模型
        set_seed(42)  # 保证公平
        model = create_model(num_classes).to(Config.DEVICE)

        # 计算参数量 (百万)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   模型参数量: {params:.2f} M")

        # 训练
        print("   开始训练...")
        trainer = ASLTrainer(model, train_loader, val_loader, num_classes=num_classes)
        # 这里的 criterion 不换成混淆损失，用基础的对比
        trainer.train()

        # 评估准确率
        print("   正在评估...")
        evaluator = ASEvaluator(model, test_loader)
        metrics = evaluator.evaluate()

        acc = metrics['accuracy']
        fps = metrics['efficiency']['fps']

        print(f"   {display_name} 结果:")
        print(f"   准确率: {acc:.4f}")
        print(f"   FPS: {fps:.2f}")

        # 记录数据
        results['names'].append(display_name)
        results['accuracy'].append(acc)
        results['params'].append(params)
        results['fps'].append(fps)

        # 释放显存
        del model
        del trainer
        torch.cuda.empty_cache()

    # 4. 恢复原始配置
    Config.MAX_EPOCHS = original_epochs

    # 5. 调用绘图函数
    plot_results(results)


def plot_results(results):
    print("\n正在根据真实数据生成对比图...")

    names = results['names']
    accs = results['accuracy']
    params = results['params']
    fps = results['fps']

    colors = ['#808080', '#87CEEB', '#FF4500']  # 灰、蓝、红(我们)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 1. 准确率
    ax1 = axs[0, 0]
    bars = ax1.bar(names, accs, color=colors)
    ax1.set_title('真实训练准确率对比', fontsize=14)
    ax1.set_ylim(0, 1.05)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.3f}', ha='center',
                 va='bottom')

    # 2. 参数量
    ax2 = axs[0, 1]
    bars = ax2.bar(names, params, color=colors)
    ax2.set_title('模型参数量对比 (M)', fontsize=14)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}M', ha='center',
                 va='bottom')

    # 3. FPS
    ax3 = axs[1, 0]
    bars = ax3.bar(names, fps, color=colors)
    ax3.set_title('推理速度对比 (FPS)', fontsize=14)
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}', ha='center',
                 va='bottom')

    # 4. 气泡图
    ax4 = axs[1, 1]
    sizes = [f * 10 for f in fps]
    ax4.scatter(params, accs, s=sizes, c=colors, alpha=0.6, edgecolors='black')
    ax4.set_title('效率-准确率综合对比 (气泡大小=FPS)', fontsize=14)
    ax4.set_xlabel('参数量 (M)')
    ax4.set_ylabel('准确率')
    ax4.grid(True, linestyle='--', alpha=0.3)

    # 添加标签
    for i, name in enumerate(names):
        ax4.annotate(name, (params[i], accs[i]), xytext=(0, 10), textcoords='offset points', ha='center')

    plt.tight_layout()
    plt.savefig('real_benchmark_comparison.png')
    print("✅ 对比图已保存: real_benchmark_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_benchmark()