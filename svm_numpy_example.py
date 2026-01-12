import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class SVM:
    """
    支持向量机(SVM)的NumPy实现
    使用梯度下降优化，支持线性核和RBF核
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, 
                 kernel='linear', gamma=0.1, verbose=False):
        """
        参数:
            learning_rate: 学习率
            lambda_param: 正则化参数（C = 1/lambda）
            n_iterations: 迭代次数
            kernel: 'linear' 或 'rbf'
            gamma: RBF核参数
            verbose: 是否打印训练过程
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma
        self.verbose = verbose
        
        self.w = None  # 权重向量
        self.b = None  # 偏置
        self.losses = []
        
        # 用于核方法
        self.X_train = None
        self.alpha = None  # 对偶系数
        self.support_vectors = None
        self.support_vector_labels = None
    
    def _linear_kernel(self, x1, x2):
        """线性核: K(x1, x2) = x1^T * x2"""
        return np.dot(x1, x2.T)
    
    def _rbf_kernel(self, x1, x2):
        """
        RBF核(高斯核): K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        """
        if len(x1.shape) == 1:
            x1 = x1.reshape(1, -1)
        if len(x2.shape) == 1:
            x2 = x2.reshape(1, -1)
        
        # 计算欧氏距离的平方
        distances = np.sum(x1**2, axis=1, keepdims=True) + \
                    np.sum(x2**2, axis=1) - \
                    2 * np.dot(x1, x2.T)
        
        return np.exp(-self.gamma * distances)
    
    def _compute_kernel_matrix(self, X1, X2):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError("kernel must be 'linear' or 'rbf'")
    
    def fit(self, X, y):
        """
        训练SVM模型
        
        参数:
            X: 训练数据，shape=(n_samples, n_features)
            y: 标签，{-1, 1}
        """
        # 确保标签是-1和1
        y = np.where(y <= 0, -1, 1).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.kernel == 'linear':
            self._fit_linear(X, y, n_samples, n_features)
        else:
            self._fit_kernel(X, y, n_samples)
        
        return self
    
    def _fit_linear(self, X, y, n_samples, n_features):
        """
        使用梯度下降训练线性SVM
        
        目标函数: min 1/2 ||w||^2 + C * Σ max(0, 1 - y_i * (w^T * x_i + b))
        
        梯度:
            当 y_i * (w^T * x_i + b) >= 1: ∂L/∂w = λ*w
            当 y_i * (w^T * x_i + b) < 1:  ∂L/∂w = λ*w - y_i*x_i
        """
        # 初始化参数
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        # 梯度下降
        for iteration in range(self.n_iterations):
            total_loss = 0
            
            for idx in range(n_samples):
                x_i = X[idx:idx+1].T
                y_i = y[idx:idx+1]
                
                # 计算判决值
                decision = np.dot(self.w.T, x_i) + self.b
                
                # Hinge Loss条件
                condition = y_i * decision >= 1
                
                if condition:
                    # 正确分类且在边界外
                    dw = self.lambda_param * self.w
                    db = 0
                    loss = 0
                else:
                    # 违反边界或分类错误
                    dw = self.lambda_param * self.w - np.dot(x_i, y_i.T)
                    db = -y_i[0, 0]
                    loss = 1 - y_i[0, 0] * decision[0, 0]
                
                # 更新参数
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
                total_loss += loss
            
            # 正则化损失
            reg_loss = 0.5 * self.lambda_param * np.sum(self.w ** 2)
            total_loss = reg_loss + total_loss / n_samples
            self.losses.append(total_loss)
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.4f}")
    
    def _fit_kernel(self, X, y, n_samples):
        """
        使用核方法训练SVM（简化版对偶形式）
        """
        self.X_train = X
        self.alpha = np.zeros((n_samples, 1))
        self.b = 0
        
        # 计算核矩阵
        K = self._compute_kernel_matrix(X, X)
        
        # 梯度下降优化对偶问题
        for iteration in range(self.n_iterations):
            total_loss = 0
            
            for idx in range(n_samples):
                # 计算决策值
                decision = np.sum(self.alpha * y * K[:, idx:idx+1]) + self.b
                
                # Hinge Loss条件
                condition = y[idx] * decision >= 1
                
                if not condition:
                    # 更新alpha
                    self.alpha[idx] += self.lr * (1 - y[idx] * decision)
                    # 限制alpha为非负
                    self.alpha[idx] = max(0, self.alpha[idx])
                    
                    # 更新偏置
                    self.b += self.lr * y[idx]
                    
                    loss = 1 - y[idx, 0] * decision
                    total_loss += loss
            
            # 正则化损失
            reg_loss = 0.5 * np.sum(self.alpha * y * np.dot(K, self.alpha * y))
            total_loss = reg_loss + total_loss / n_samples
            self.losses.append(total_loss)
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.4f}")
        
        # 找出支持向量（alpha > 0的样本）
        sv_indices = np.where(self.alpha.ravel() > 1e-5)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]
        
        if self.verbose:
            print(f"\n找到 {len(sv_indices)} 个支持向量")
    
    def predict(self, X):
        """
        预测
        
        返回:
            预测标签 {-1, 1}
        """
        if self.kernel == 'linear':
            decision = np.dot(X, self.w) + self.b
        else:
            # 使用支持向量进行预测
            K = self._compute_kernel_matrix(X, self.support_vectors)
            decision = np.sum(self.alpha.T * self.support_vector_labels.T * K.T, axis=0, keepdims=True).T + self.b
        
        return np.sign(decision)
    
    def decision_function(self, X):
        """
        计算决策函数值（到超平面的距离）
        """
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.b
        else:
            K = self._compute_kernel_matrix(X, self.support_vectors)
            return np.sum(self.alpha.T * self.support_vector_labels.T * K.T, axis=0, keepdims=True).T + self.b
    
    def score(self, X, y):
        """计算准确率"""
        y = np.where(y <= 0, -1, 1).reshape(-1, 1)
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


def generate_binary_classification_data(n_samples=200, noise=0.2, random_state=42):
    """
    生成二分类数据
    """
    np.random.seed(random_state)
    
    # 生成两个类别
    n_per_class = n_samples // 2
    
    # 类别1：中心在(-1, -1)
    X1 = np.random.randn(n_per_class, 2) * noise + np.array([-1, -1])
    y1 = np.ones((n_per_class, 1))
    
    # 类别2：中心在(1, 1)
    X2 = np.random.randn(n_per_class, 2) * noise + np.array([1, 1])
    y2 = -np.ones((n_per_class, 1))
    
    # 合并
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    
    # 打乱
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_nonlinear_data(n_samples=200, noise=0.15, random_state=42):
    """
    生成非线性可分数据（XOR问题）
    """
    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # 生成圆形分布的数据
    theta = np.random.rand(n_per_class) * 2 * np.pi
    r1 = np.random.randn(n_per_class) * noise + 1
    r2 = np.random.randn(n_per_class) * noise + 2.5
    
    X1 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
    X2 = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)])
    
    y1 = np.ones((n_per_class, 1))
    y2 = -np.ones((n_per_class, 1))
    
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    
    # 打乱
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    """
    绘制决策边界
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制等高线
    plt.contourf(xx, yy, Z, levels=[-np.inf, -1, 0, 1, np.inf],
                 colors=['#FFAAAA', '#FFDDDD', '#DDDDFF', '#AAAAFF'], alpha=0.8)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
                colors=['red', 'black', 'blue'], 
                linewidths=[1, 2, 1],
                linestyles=['--', '-', '--'])
    
    # 绘制数据点
    y_flat = y.ravel()
    plt.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
                c='blue', marker='o', s=50, edgecolors='black', label='类别 +1')
    plt.scatter(X[y_flat == -1, 0], X[y_flat == -1, 1], 
                c='red', marker='s', s=50, edgecolors='black', label='类别 -1')
    
    # 标记支持向量
    if model.kernel == 'linear':
        # 对于线性核，找到离超平面最近的点作为支持向量的近似
        distances = np.abs(model.decision_function(X))
        sv_threshold = np.percentile(distances, 10)
        sv_mask = distances.ravel() <= sv_threshold
        plt.scatter(X[sv_mask, 0], X[sv_mask, 1], 
                   s=200, facecolors='none', edgecolors='green', 
                   linewidths=2, label='支持向量(近似)')
    else:
        if model.support_vectors is not None:
            plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1],
                       s=200, facecolors='none', edgecolors='green',
                       linewidths=2, label='支持向量')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)


def example_linear_svm():
    """
    示例1：线性可分数据
    """
    print("=" * 70)
    print("示例1：线性SVM（线性可分数据）")
    print("=" * 70)
    
    # 生成线性可分数据
    X, y = generate_binary_classification_data(n_samples=200, noise=0.3)
    
    # 划分训练集和测试集
    split_idx = 160
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 训练线性SVM
    print("\n训练线性SVM...")
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000, 
              kernel='linear', verbose=True)
    svm.fit(X_train, y_train)
    
    # 评估
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"\n学到的参数:")
    print(f"权重 w: {svm.w.ravel()}")
    print(f"偏置 b: {svm.b:.4f}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary(svm, X, y, "线性SVM决策边界")
    
    # 损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(svm.losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    
    # 决策函数值分布
    plt.subplot(1, 3, 3)
    decision_values = svm.decision_function(X)
    y_flat = y.ravel()
    plt.hist(decision_values[y_flat == 1], bins=30, alpha=0.5, label='类别 +1', color='blue')
    plt.hist(decision_values[y_flat == -1], bins=30, alpha=0.5, label='类别 -1', color='red')
    plt.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='边界 -1')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2, label='超平面')
    plt.axvline(x=1, color='blue', linestyle='--', linewidth=2, label='边界 +1')
    plt.xlabel('决策函数值')
    plt.ylabel('频数')
    plt.title('决策函数值分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_linear.png', dpi=150)
    print("\n结果图已保存: svm_linear.png")
    plt.show()


def example_rbf_svm():
    """
    示例2：非线性可分数据（RBF核）
    """
    print("\n" + "=" * 70)
    print("示例2：RBF核SVM（非线性可分数据）")
    print("=" * 70)
    
    # 生成非线性数据
    X, y = generate_nonlinear_data(n_samples=200, noise=0.15)
    
    # 划分数据集
    split_idx = 160
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 训练RBF核SVM
    print("\n训练RBF核SVM...")
    svm_rbf = SVM(learning_rate=0.01, lambda_param=0.01, n_iterations=1000,
                  kernel='rbf', gamma=0.5, verbose=True)
    svm_rbf.fit(X_train, y_train)
    
    # 评估
    train_acc = svm_rbf.score(X_train, y_train)
    test_acc = svm_rbf.score(X_test, y_test)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # RBF核决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary(svm_rbf, X, y, "RBF核SVM决策边界")
    
    # 损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(svm_rbf.losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    
    # 对比线性核
    print("\n对比：训练线性核SVM...")
    svm_linear = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000,
                     kernel='linear', verbose=False)
    svm_linear.fit(X_train, y_train)
    linear_acc = svm_linear.score(X_test, y_test)
    
    plt.subplot(1, 3, 3)
    plot_decision_boundary(svm_linear, X, y, f"线性核SVM (准确率: {linear_acc:.2f})")
    
    plt.tight_layout()
    plt.savefig('svm_rbf.png', dpi=150)
    print(f"\n线性核测试准确率: {linear_acc:.4f}")
    print(f"RBF核测试准确率: {test_acc:.4f}")
    print("\n结果图已保存: svm_rbf.png")
    plt.show()


def compare_parameters():
    """
    示例3：超参数对比
    """
    print("\n" + "=" * 70)
    print("示例3：不同超参数对比")
    print("=" * 70)
    
    # 生成数据
    X, y = generate_binary_classification_data(n_samples=200, noise=0.4)
    
    # 测试不同的C值（C = 1/lambda）
    lambda_values = [0.001, 0.01, 0.1, 1.0]
    
    plt.figure(figsize=(15, 10))
    
    for idx, lambda_param in enumerate(lambda_values):
        C = 1 / lambda_param
        print(f"\n训练SVM with C={C:.2f} (lambda={lambda_param})...")
        
        svm = SVM(learning_rate=0.001, lambda_param=lambda_param, 
                  n_iterations=1000, kernel='linear', verbose=False)
        svm.fit(X, y)
        
        accuracy = svm.score(X, y)
        
        plt.subplot(2, 2, idx + 1)
        plot_decision_boundary(svm, X, y, 
                              f"C={C:.2f}, 准确率={accuracy:.3f}")
    
    plt.tight_layout()
    plt.savefig('svm_parameters.png', dpi=150)
    print("\n结果图已保存: svm_parameters.png")
    plt.show()


def mathematical_explanation():
    """
    SVM数学原理说明
    """
    explanation = """
═══════════════════════════════════════════════════════════════════════
支持向量机(SVM)数学原理
═══════════════════════════════════════════════════════════════════════

1. 核心思想
   找到一个最优超平面，最大化两类样本之间的间隔(margin)

2. 线性SVM

   原始问题（硬间隔）:
   
   min  1/2 ||w||²
   s.t. y_i(w^T·x_i + b) ≥ 1, ∀i
   
   软间隔SVM（允许误分类）:
   
   min  1/2 ||w||² + C·Σξ_i
   s.t. y_i(w^T·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0
   
   其中 C 是正则化参数，ξ_i 是松弛变量

3. Hinge Loss表示

   L(w, b) = λ·||w||² + (1/n)·Σ max(0, 1 - y_i·(w^T·x_i + b))
   
   其中 λ = 1/(2C)

4. 梯度

   当 y_i·(w^T·x_i + b) ≥ 1:  ∂L/∂w = λ·w
   当 y_i·(w^T·x_i + b) < 1:   ∂L/∂w = λ·w - y_i·x_i

5. 支持向量

   满足 y_i·(w^T·x_i + b) = 1 的样本点
   只有支持向量对决策边界有影响

6. 核技巧

   对于非线性问题，使用核函数将数据映射到高维空间
   
   线性核: K(x_i, x_j) = x_i^T·x_j
   
   RBF核(高斯核): K(x_i, x_j) = exp(-γ·||x_i - x_j||²)
   
   多项式核: K(x_i, x_j) = (x_i^T·x_j + c)^d

7. 对偶问题

   max  Σα_i - (1/2)·Σ Σ α_i·α_j·y_i·y_j·K(x_i, x_j)
   s.t. 0 ≤ α_i ≤ C, Σ α_i·y_i = 0
   
   决策函数: f(x) = sign(Σ α_i·y_i·K(x_i, x) + b)

8. 超参数

   - C: 正则化强度（C越大，容忍误分类越少，可能过拟合）
   - γ: RBF核参数（γ越大，决策边界越复杂）
   
═══════════════════════════════════════════════════════════════════════
"""
    print(explanation)


if __name__ == "__main__":
    # 打印数学原理
    print("=" * 70)
    print("支持向量机(SVM) - NumPy实现")
    print("=" * 70)
    mathematical_explanation()
    
    # 运行示例
    example_linear_svm()
    example_rbf_svm()
    compare_parameters()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
