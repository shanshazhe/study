"""
逻辑回归的损失函数详解

本文件详细说明逻辑回归中常用的损失函数及其数学原理
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. 二元交叉熵损失 (Binary Cross-Entropy Loss) - 最常用
# =============================================================================
"""
这是逻辑回归的标准损失函数，也称为对数损失(Log Loss)

数学公式:
    Loss = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
其中:
    - m: 样本数量
    - y: 真实标签 (0 或 1)
    - ŷ: 预测概率 sigmoid(w^T*x + b)
    
单个样本的损失:
    - 当 y=1 时: Loss = -log(ŷ)      → ŷ越接近1，损失越小
    - 当 y=0 时: Loss = -log(1-ŷ)    → ŷ越接近0，损失越小

为什么用交叉熵？
    1. 基于最大似然估计推导而来（统计学基础）
    2. 凸函数，容易优化（梯度下降保证收敛到全局最优）
    3. 梯度形式简洁: ∂L/∂w = 1/m * X^T * (ŷ - y)
    4. 概率解释清晰
"""

def binary_crossentropy_loss(y_true, y_pred):
    """二元交叉熵损失"""
    m = len(y_true)
    epsilon = 1e-15  # 防止log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# =============================================================================
# 2. 均方误差损失 (Mean Squared Error - MSE)
# =============================================================================
"""
虽然不是逻辑回归的标准损失，但也可以使用

数学公式:
    Loss = 1/(2*m) * Σ(ŷ - y)²
    
其中:
    - ŷ: sigmoid(w^T*x + b)
    - y: 真实标签 (0 或 1)

梯度公式:
    ∂L/∂w = 1/m * X^T * [(ŷ - y) * ŷ * (1 - ŷ)]
    
注意: 梯度中包含sigmoid的导数项 ŷ*(1-ŷ)

缺点:
    1. 非凸函数，可能陷入局部最优
    2. 当预测非常错误时，梯度会变小（梯度消失）
    3. 没有概率解释
    
优点:
    1. 计算简单
    2. 直观易懂
"""

def mse_loss(y_true, y_pred):
    """均方误差损失"""
    m = len(y_true)
    loss = 1/(2*m) * np.sum((y_pred - y_true) ** 2)
    return loss


# =============================================================================
# 3. Hinge Loss（合页损失）
# =============================================================================
"""
来自支持向量机(SVM)，也可用于逻辑回归

数学公式:
    Loss = 1/m * Σ max(0, 1 - y'*z)
    
其中:
    - y': 标签转换为 {-1, +1}
    - z = w^T*x + b (未经过sigmoid)

特点:
    1. 只关注分类边界附近的样本
    2. 正确且置信度高的样本(y'*z > 1)损失为0
    3. 分类错误的样本有线性惩罚
    
梯度:
    当 y'*z < 1 时: ∂L/∂w = -1/m * X^T * y'
    当 y'*z ≥ 1 时: ∂L/∂w = 0
"""

def hinge_loss(y_true, z):
    """Hinge损失"""
    m = len(y_true)
    y_transformed = 2 * y_true - 1  # {0,1} → {-1,1}
    loss = 1/m * np.sum(np.maximum(0, 1 - y_transformed * z))
    return loss


# =============================================================================
# 4. 指数损失 (Exponential Loss)
# =============================================================================
"""
常用于AdaBoost算法

数学公式:
    Loss = 1/m * Σ exp(-y'*z)
    
其中:
    - y': 标签 {-1, +1}
    - z = w^T*x + b

特点:
    1. 对离群点非常敏感
    2. 指数增长的惩罚
    3. 不常用于标准逻辑回归
"""

def exponential_loss(y_true, z):
    """指数损失"""
    m = len(y_true)
    y_transformed = 2 * y_true - 1  # {0,1} → {-1,1}
    loss = 1/m * np.sum(np.exp(-y_transformed * z))
    return loss


# =============================================================================
# 5. 感知机损失 (Perceptron Loss)
# =============================================================================
"""
最简单的线性分类损失

数学公式:
    Loss = 1/m * Σ max(0, -y'*z)
    
其中:
    - y': 标签 {-1, +1}
    - z = w^T*x + b

特点:
    1. 只惩罚分类错误的样本
    2. 正确分类的样本损失为0（无论置信度）
    3. 比Hinge Loss更激进
"""

def perceptron_loss(y_true, z):
    """感知机损失"""
    m = len(y_true)
    y_transformed = 2 * y_true - 1  # {0,1} → {-1,1}
    loss = 1/m * np.sum(np.maximum(0, -y_transformed * z))
    return loss


# =============================================================================
# 可视化对比所有损失函数
# =============================================================================

def visualize_loss_functions():
    """
    可视化不同损失函数的形状
    """
    # 假设真实标签 y=1，横轴是模型输出z或预测概率
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 交叉熵损失 (横轴是预测概率)
    ax1 = axes[0, 0]
    y_pred = np.linspace(0.01, 0.99, 100)
    ce_loss_y1 = -np.log(y_pred)  # y=1时的损失
    ce_loss_y0 = -np.log(1 - y_pred)  # y=0时的损失
    
    ax1.plot(y_pred, ce_loss_y1, 'b-', linewidth=2, label='y=1')
    ax1.plot(y_pred, ce_loss_y0, 'r-', linewidth=2, label='y=0')
    ax1.set_xlabel('预测概率 ŷ')
    ax1.set_ylabel('Loss')
    ax1.set_title('1. 交叉熵损失 (标准)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 5])
    
    # 2. MSE损失 (横轴是预测概率)
    ax2 = axes[0, 1]
    mse_y1 = (y_pred - 1) ** 2  # y=1时的损失
    mse_y0 = y_pred ** 2  # y=0时的损失
    
    ax2.plot(y_pred, mse_y1, 'b-', linewidth=2, label='y=1')
    ax2.plot(y_pred, mse_y0, 'r-', linewidth=2, label='y=0')
    ax2.set_xlabel('预测概率 ŷ')
    ax2.set_ylabel('Loss')
    ax2.set_title('2. 均方误差损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hinge损失 (横轴是原始输出z)
    ax3 = axes[1, 0]
    z = np.linspace(-3, 3, 100)
    hinge_y1 = np.maximum(0, 1 - z)  # y=1时的损失 (y'=1)
    hinge_y0 = np.maximum(0, 1 + z)  # y=0时的损失 (y'=-1)
    
    ax3.plot(z, hinge_y1, 'b-', linewidth=2, label='y=1')
    ax3.plot(z, hinge_y0, 'r-', linewidth=2, label='y=0')
    ax3.set_xlabel('模型输出 z')
    ax3.set_ylabel('Loss')
    ax3.set_title('3. Hinge损失 (SVM-like)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 所有损失对比 (y=1的情况)
    ax4 = axes[1, 1]
    z_range = np.linspace(-3, 3, 100)
    sigmoid_pred = 1 / (1 + np.exp(-z_range))
    
    ce_loss = -np.log(np.clip(sigmoid_pred, 1e-15, 1-1e-15))
    mse_loss_vals = (sigmoid_pred - 1) ** 2
    hinge_loss_vals = np.maximum(0, 1 - z_range)
    perceptron_loss_vals = np.maximum(0, -z_range)
    
    ax4.plot(z_range, ce_loss, 'b-', linewidth=2, label='Cross-Entropy')
    ax4.plot(z_range, mse_loss_vals, 'g-', linewidth=2, label='MSE')
    ax4.plot(z_range, hinge_loss_vals, 'r-', linewidth=2, label='Hinge')
    ax4.plot(z_range, perceptron_loss_vals, 'm-', linewidth=2, label='Perceptron')
    ax4.set_xlabel('模型输出 z')
    ax4.set_ylabel('Loss')
    ax4.set_title('4. 损失函数对比 (y=1)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 3])
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_loss_functions.png', dpi=150, bbox_inches='tight')
    print("损失函数对比图已保存: logistic_regression_loss_functions.png")
    plt.show()


def print_summary():
    """
    打印损失函数总结
    """
    print("=" * 80)
    print("逻辑回归损失函数总结")
    print("=" * 80)
    print()
    print("1. 二元交叉熵 (Binary Cross-Entropy) ⭐ 最常用")
    print("   公式: -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]")
    print("   特点: 凸函数、概率解释、梯度稳定")
    print("   应用: 标准逻辑回归")
    print()
    print("2. 均方误差 (MSE)")
    print("   公式: 1/(2*m) * Σ(ŷ - y)²")
    print("   特点: 非凸、可能梯度消失")
    print("   应用: 回归问题，分类效果较差")
    print()
    print("3. Hinge Loss")
    print("   公式: 1/m * Σ max(0, 1 - y'*z)")
    print("   特点: 关注边界、稀疏解")
    print("   应用: SVM、鲁棒分类")
    print()
    print("4. 指数损失 (Exponential)")
    print("   公式: 1/m * Σ exp(-y'*z)")
    print("   特点: 对离群点敏感")
    print("   应用: AdaBoost")
    print()
    print("5. 感知机损失 (Perceptron)")
    print("   公式: 1/m * Σ max(0, -y'*z)")
    print("   特点: 只惩罚错误样本")
    print("   应用: 感知机算法")
    print()
    print("=" * 80)
    print("推荐：二元交叉熵是逻辑回归的标准和最佳选择！")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
    print("\n正在生成损失函数可视化对比图...\n")
    visualize_loss_functions()
