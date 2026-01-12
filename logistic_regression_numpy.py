import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    使用NumPy实现的逻辑回归模型
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数: 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def initialize_parameters(self, n_features):
        """
        初始化权重和偏置
        """
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
    
    def compute_loss(self, y_true, y_pred):
        """
        计算二元交叉熵损失
        Loss = -1/m * Σ(y*log(ŷ) + (1-y)*log(1-ŷ))
        """
        m = y_true.shape[0]
        epsilon = 1e-15  # 避免log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """
        训练逻辑回归模型

        参数:
            X: 训练数据，shape=(m, n) m个样本，n个特征
            y: 标签，shape=(m, 1)
        """
        m, n = X.shape
        
        # 初始化参数
        self.initialize_parameters(n)
        
        # 梯度下降
        for i in range(self.num_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # 反向传播：计算梯度
            dw = 1/m * np.dot(X.T, (y_pred - y))
            db = 1/m * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 打印训练过程
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict_proba(self, X):
        """
        预测概率
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别（0或1）
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)
    
    def accuracy(self, y_true, y_pred):
        """
        计算准确率
        """
        return np.mean(y_true == y_pred)


def generate_dataset(n_samples=200, n_features=2, noise=0.2):
    """
    生成二分类数据集
    """
    np.random.seed(42)
    
    # 生成两个类别的数据
    n_samples_per_class = n_samples // 2
    
    # 类别0：均值为[-1, -1]
    X0 = np.random.randn(n_samples_per_class, n_features) * noise + np.array([-1, -1])
    y0 = np.zeros((n_samples_per_class, 1))
    
    # 类别1：均值为[1, 1]
    X1 = np.random.randn(n_samples_per_class, n_features) * noise + np.array([1, 1])
    y1 = np.ones((n_samples_per_class, 1))
    
    # 合并数据
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundary(model, X, y):
    """
    绘制决策边界
    """
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 创建网格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格上的每个点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制等高线
    plt.figure(figsize=(12, 4))
    
    # 子图1：决策边界
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('逻辑回归决策边界')
    plt.colorbar()
    
    # 子图2：损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_results.png', dpi=150)
    print("\n结果图已保存为: logistic_regression_results.png")
    plt.show()


def train_example():
    """
    完整的训练示例
    """
    print("=" * 60)
    print("使用NumPy实现逻辑回归")
    print("=" * 60)
    
    # 生成数据集
    print("\n1. 生成数据集...")
    X, y = generate_dataset(n_samples=200, n_features=2, noise=0.5)
    
    # 划分训练集和测试集
    split_ratio = 0.8
    split_index = int(X.shape[0] * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"   训练集大小: {X_train.shape[0]}")
    print(f"   测试集大小: {X_test.shape[0]}")
    
    # 创建并训练模型
    print("\n2. 训练逻辑回归模型...")
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000, verbose=True)
    model.fit(X_train, y_train)
    
    # 在训练集上评估
    print("\n3. 模型评估...")
    y_train_pred = model.predict(X_train)
    train_accuracy = model.accuracy(y_train, y_train_pred)
    print(f"   训练集准确率: {train_accuracy:.4f}")
    
    # 在测试集上评估
    y_test_pred = model.predict(X_test)
    test_accuracy = model.accuracy(y_test, y_test_pred)
    print(f"   测试集准确率: {test_accuracy:.4f}")
    
    # 打印模型参数
    print("\n4. 模型参数:")
    print(f"   权重: {model.weights.ravel()}")
    print(f"   偏置: {model.bias:.4f}")
    
    # 绘制决策边界
    print("\n5. 绘制决策边界...")
    plot_decision_boundary(model, X, y)
    
    # 示例预测
    print("\n6. 示例预测:")
    test_samples = np.array([[-1, -1], [1, 1], [0, 0]])
    for sample in test_samples:
        prob = model.predict_proba(sample.reshape(1, -1))[0, 0]
        pred = model.predict(sample.reshape(1, -1))[0, 0]
        print(f"   样本 {sample}: 概率={prob:.4f}, 预测类别={pred}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    train_example()
