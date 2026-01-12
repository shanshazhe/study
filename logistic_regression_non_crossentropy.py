import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionMSE:
    """
    使用均方误差(MSE)作为损失函数的逻辑回归模型
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
        计算均方误差损失 (MSE)
        Loss = 1/(2*m) * Σ(ŷ - y)²
        """
        m = y_true.shape[0]
        loss = 1/(2*m) * np.sum((y_pred - y_true) ** 2)
        return loss
    
    def fit(self, X, y):
        """
        训练逻辑回归模型（使用MSE损失）
        
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
            # MSE的梯度: dL/dw = 1/m * X^T * ((ŷ - y) * ŷ * (1 - ŷ))
            error = y_pred - y
            sigmoid_derivative = y_pred * (1 - y_pred)
            gradient = error * sigmoid_derivative
            
            dw = 1/m * np.dot(X.T, gradient)
            db = 1/m * np.sum(1)
            
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


class LogisticRegressionHinge:
    """
    使用Hinge Loss的逻辑回归模型（类似SVM）
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
        Sigmoid激活函数
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def initialize_parameters(self, n_features):
        """
        初始化权重和偏置
        """
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
    
    def compute_loss(self, y_true, z):
        """
        计算Hinge Loss
        Loss = 1/m * Σ max(0, 1 - y * z)
        其中 y ∈ {-1, 1}，z = w^T * x + b
        """
        m = y_true.shape[0]
        # 将y从{0,1}转换为{-1,1}
        y_transformed = 2 * y_true - 1
        hinge = np.maximum(0, 1 - y_transformed * z)
        loss = 1/m * np.sum(hinge)
        return loss
    
    def fit(self, X, y):
        """
        训练模型（使用Hinge Loss）
        """
        m, n = X.shape
        
        # 初始化参数
        self.initialize_parameters(n)
        
        # 梯度下降
        for i in range(self.num_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            
            # 计算损失
            loss = self.compute_loss(y, z)
            self.losses.append(loss)
            
            # 反向传播：计算梯度
            # 将y从{0,1}转换为{-1,1}
            y_transformed = 2 * y - 1
            
            # Hinge loss的梯度
            condition = (y_transformed * z < 1).astype(float)
            dw = -1/m * np.dot(X.T, condition * y_transformed)
            db = -1/m * np.sum(condition * y_transformed)
            
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
        预测类别
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
    
    n_samples_per_class = n_samples // 2
    
    # 类别0
    X0 = np.random.randn(n_samples_per_class, n_features) * noise + np.array([-1, -1])
    y0 = np.zeros((n_samples_per_class, 1))
    
    # 类别1
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


def compare_models():
    """
    比较不同损失函数的逻辑回归模型
    """
    print("=" * 70)
    print("比较不同损失函数的逻辑回归模型")
    print("=" * 70)
    
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
    
    # 训练MSE模型
    print("\n2. 训练MSE损失函数的逻辑回归...")
    model_mse = LogisticRegressionMSE(learning_rate=0.5, num_iterations=1000, verbose=True)
    model_mse.fit(X_train, y_train)
    
    y_train_pred_mse = model_mse.predict(X_train)
    train_acc_mse = model_mse.accuracy(y_train, y_train_pred_mse)
    y_test_pred_mse = model_mse.predict(X_test)
    test_acc_mse = model_mse.accuracy(y_test, y_test_pred_mse)
    
    print(f"   训练集准确率: {train_acc_mse:.4f}")
    print(f"   测试集准确率: {test_acc_mse:.4f}")
    
    # 训练Hinge Loss模型
    print("\n3. 训练Hinge Loss的逻辑回归...")
    model_hinge = LogisticRegressionHinge(learning_rate=0.1, num_iterations=1000, verbose=True)
    model_hinge.fit(X_train, y_train)
    
    y_train_pred_hinge = model_hinge.predict(X_train)
    train_acc_hinge = model_hinge.accuracy(y_train, y_train_pred_hinge)
    y_test_pred_hinge = model_hinge.predict(X_test)
    test_acc_hinge = model_hinge.accuracy(y_test, y_test_pred_hinge)
    
    print(f"   训练集准确率: {train_acc_hinge:.4f}")
    print(f"   测试集准确率: {test_acc_hinge:.4f}")
    
    # 可视化比较
    print("\n4. 绘制损失曲线对比...")
    plt.figure(figsize=(15, 5))
    
    # MSE模型决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary_subplot(model_mse, X, y, "MSE Loss")
    
    # Hinge Loss模型决策边界
    plt.subplot(1, 3, 2)
    plot_decision_boundary_subplot(model_hinge, X, y, "Hinge Loss")
    
    # 损失曲线对比
    plt.subplot(1, 3, 3)
    plt.plot(model_mse.losses, label='MSE Loss', linewidth=2)
    plt.plot(model_hinge.losses, label='Hinge Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('损失函数对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_comparison.png', dpi=150)
    print("   结果图已保存为: logistic_regression_comparison.png")
    plt.show()
    
    # 总结
    print("\n" + "=" * 70)
    print("模型性能总结:")
    print(f"   MSE模型 - 测试准确率: {test_acc_mse:.4f}")
    print(f"   Hinge Loss模型 - 测试准确率: {test_acc_hinge:.4f}")
    print("=" * 70)


def plot_decision_boundary_subplot(model, X, y, title):
    """
    在子图中绘制决策边界
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)


if __name__ == "__main__":
    compare_models()
