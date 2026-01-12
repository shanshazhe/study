import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    使用NumPy实现的线性回归模型
    支持梯度下降和正规方程两种求解方法
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, method='gradient_descent', verbose=False):
        """
        参数:
            learning_rate: 学习率
            num_iterations: 迭代次数
            method: 'gradient_descent' 或 'normal_equation'
            verbose: 是否打印训练过程
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.method = method
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
    
    def initialize_parameters(self, n_features):
        """初始化权重和偏置"""
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
    
    def fit_gradient_descent(self, X, y):
        """
        使用梯度下降训练模型
        
        参数:
            X: 训练数据，shape=(m, n)
            y: 标签，shape=(m, 1)
        """
        m, n = X.shape
        self.initialize_parameters(n)
        
        for i in range(self.num_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias
            
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
    
    def fit_normal_equation(self, X, y):
        """
        使用正规方程直接求解
        
        正规方程: θ = (X^T * X)^(-1) * X^T * y
        其中 θ = [bias; weights]
        """
        # 在X前添加一列1（用于偏置项）
        m = X.shape[0]
        X_b = np.hstack([np.ones((m, 1)), X])
        
        # 计算正规方程
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # 提取偏置和权重
        self.bias = theta[0, 0]
        self.weights = theta[1:, :]
        
        # 计算最终损失
        y_pred = np.dot(X, self.weights) + self.bias
        loss = self.compute_loss(y, y_pred)
        self.losses.append(loss)
        
        if self.verbose:
            print(f"Normal Equation - Final Loss: {loss:.4f}")
    
    def fit(self, X, y):
        """
        训练模型
        """
        if self.method == 'gradient_descent':
            self.fit_gradient_descent(X, y)
        elif self.method == 'normal_equation':
            self.fit_normal_equation(X, y)
        else:
            raise ValueError("method must be 'gradient_descent' or 'normal_equation'")
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias
    
    def r2_score(self, y_true, y_pred):
        """
        计算R²决定系数
        R² = 1 - (SS_res / SS_tot)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def mse(self, y_true, y_pred):
        """计算均方误差"""
        return np.mean((y_true - y_pred) ** 2)
    
    def mae(self, y_true, y_pred):
        """计算平均绝对误差"""
        return np.mean(np.abs(y_true - y_pred))


def generate_linear_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """
    生成线性回归数据
    y = X * true_weights + true_bias + noise
    """
    np.random.seed(random_state)
    
    # 生成输入特征
    X = 2 * np.random.rand(n_samples, n_features)
    
    # 真实的权重和偏置
    true_weights = np.random.randn(n_features, 1) * 10
    true_bias = np.random.randn() * 5
    
    # 生成标签（带噪声）
    y = np.dot(X, true_weights) + true_bias + noise * np.random.randn(n_samples, 1)
    
    return X, y, true_weights, true_bias


def example_1d_linear_regression():
    """
    示例1：一维线性回归（单特征）
    """
    print("=" * 70)
    print("示例1：一维线性回归")
    print("=" * 70)
    
    # 生成数据
    X, y, true_weights, true_bias = generate_linear_data(n_samples=100, n_features=1, noise=5)
    print(f"\n真实参数: weight={true_weights[0,0]:.2f}, bias={true_bias:.2f}")
    
    # 划分训练集和测试集
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 方法1：梯度下降
    print("\n--- 方法1：梯度下降 ---")
    model_gd = LinearRegression(learning_rate=0.1, num_iterations=1000, 
                                 method='gradient_descent', verbose=True)
    model_gd.fit(X_train, y_train)
    
    y_pred_gd = model_gd.predict(X_test)
    print(f"学到的参数: weight={model_gd.weights[0,0]:.2f}, bias={model_gd.bias:.2f}")
    print(f"测试集 R²: {model_gd.r2_score(y_test, y_pred_gd):.4f}")
    print(f"测试集 MSE: {model_gd.mse(y_test, y_pred_gd):.4f}")
    
    # 方法2：正规方程
    print("\n--- 方法2：正规方程 ---")
    model_ne = LinearRegression(method='normal_equation', verbose=True)
    model_ne.fit(X_train, y_train)
    
    y_pred_ne = model_ne.predict(X_test)
    print(f"学到的参数: weight={model_ne.weights[0,0]:.2f}, bias={model_ne.bias:.2f}")
    print(f"测试集 R²: {model_ne.r2_score(y_test, y_pred_ne):.4f}")
    print(f"测试集 MSE: {model_ne.mse(y_test, y_pred_ne):.4f}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 子图1：数据和拟合直线
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='测试数据')
    
    X_line = np.array([[0], [2]])
    y_line_gd = model_gd.predict(X_line)
    y_line_ne = model_ne.predict(X_line)
    
    plt.plot(X_line, y_line_gd, 'g-', linewidth=2, label='梯度下降')
    plt.plot(X_line, y_line_ne, 'b--', linewidth=2, label='正规方程')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(model_gd.losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('梯度下降损失曲线')
    plt.grid(True, alpha=0.3)
    
    # 子图3：预测vs真实
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred_gd, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='理想预测')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测vs真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_1d.png', dpi=150)
    print("\n结果图已保存: linear_regression_1d.png")
    plt.show()


def example_multi_feature_regression():
    """
    示例2：多元线性回归（多特征）
    """
    print("\n" + "=" * 70)
    print("示例2：多元线性回归（3个特征）")
    print("=" * 70)
    
    # 生成多特征数据
    X, y, true_weights, true_bias = generate_linear_data(n_samples=200, n_features=3, noise=5)
    print(f"\n真实参数:")
    print(f"  weights: {true_weights.ravel()}")
    print(f"  bias: {true_bias:.2f}")
    
    # 划分数据集
    split_idx = 160
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 训练模型
    print("\n训练中...")
    model = LinearRegression(learning_rate=0.1, num_iterations=1000, 
                            method='gradient_descent', verbose=False)
    model.fit(X_train, y_train)
    
    # 评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"\n学到的参数:")
    print(f"  weights: {model.weights.ravel()}")
    print(f"  bias: {model.bias:.2f}")
    
    print(f"\n训练集性能:")
    print(f"  R²: {model.r2_score(y_train, y_train_pred):.4f}")
    print(f"  MSE: {model.mse(y_train, y_train_pred):.4f}")
    print(f"  MAE: {model.mae(y_train, y_train_pred):.4f}")
    
    print(f"\n测试集性能:")
    print(f"  R²: {model.r2_score(y_test, y_test_pred):.4f}")
    print(f"  MSE: {model.mse(y_test, y_test_pred):.4f}")
    print(f"  MAE: {model.mae(y_test, y_test_pred):.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(model.losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    
    # 预测vs真实
    plt.subplot(1, 2, 2)
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='训练集')
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='测试集')
    
    all_y = np.vstack([y_train, y_test])
    plt.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 
             'g--', linewidth=2, label='理想预测')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测vs真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_multi.png', dpi=150)
    print("\n结果图已保存: linear_regression_multi.png")
    plt.show()


def example_polynomial_regression():
    """
    示例3：多项式回归（特征工程）
    """
    print("\n" + "=" * 70)
    print("示例3：多项式回归")
    print("=" * 70)
    
    # 生成非线性数据
    np.random.seed(42)
    X = np.linspace(0, 3, 100).reshape(-1, 1)
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5
    
    # 特征工程：添加多项式特征
    X_poly = np.hstack([X, X**2])  # [X, X²]
    
    # 训练模型
    print("\n训练多项式回归模型...")
    model = LinearRegression(learning_rate=0.01, num_iterations=2000, 
                            method='gradient_descent', verbose=False)
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    
    print(f"\n学到的参数:")
    print(f"  weight_X: {model.weights[0,0]:.4f}")
    print(f"  weight_X²: {model.weights[1,0]:.4f}")
    print(f"  bias: {model.bias:.4f}")
    print(f"\nR²: {model.r2_score(y, y_pred):.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='数据')
    plt.plot(X, y_pred, 'r-', linewidth=2, label='多项式拟合')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('多项式回归 (y = aX² + bX + c)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_polynomial.png', dpi=150)
    print("\n结果图已保存: linear_regression_polynomial.png")
    plt.show()


if __name__ == "__main__":
    # 运行所有示例
    example_1d_linear_regression()
    example_multi_feature_regression()
    example_polynomial_regression()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
