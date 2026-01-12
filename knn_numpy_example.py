import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class KNN:
    """
    K近邻算法(KNN)的NumPy实现
    支持分类和回归任务
    """
    def __init__(self, k=3, task='classification', distance_metric='euclidean', weights='uniform'):
        """
        参数:
            k: 近邻数量
            task: 'classification' 或 'regression'
            distance_metric: 'euclidean', 'manhattan', 'minkowski', 'cosine'
            weights: 'uniform' 或 'distance'（距离加权）
        """
        self.k = k
        self.task = task
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        训练KNN模型（实际上只是存储训练数据）
        
        参数:
            X: 训练数据，shape=(n_samples, n_features)
            y: 标签，shape=(n_samples,) 或 (n_samples, 1)
        """
        self.X_train = X
        self.y_train = y.reshape(-1, 1) if len(y.shape) == 1 else y
        return self
    
    def _euclidean_distance(self, x1, x2):
        """欧氏距离: sqrt(Σ(x1 - x2)²)"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def _manhattan_distance(self, x1, x2):
        """曼哈顿距离: Σ|x1 - x2|"""
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def _minkowski_distance(self, x1, x2, p=3):
        """闵可夫斯基距离: (Σ|x1 - x2|^p)^(1/p)"""
        return np.power(np.sum(np.abs(x1 - x2) ** p, axis=1), 1/p)
    
    def _cosine_distance(self, x1, x2):
        """
        余弦距离: 1 - cos(θ) = 1 - (x1·x2)/(||x1||·||x2||)
        """
        dot_product = np.sum(x1 * x2, axis=1)
        norm_x1 = np.linalg.norm(x1, axis=1)
        norm_x2 = np.linalg.norm(x2, axis=1)
        cosine_similarity = dot_product / (norm_x1 * norm_x2 + 1e-8)
        return 1 - cosine_similarity
    
    def _compute_distances(self, X_test):
        """
        计算测试样本到所有训练样本的距离
        
        返回:
            distances: shape=(n_test_samples, n_train_samples)
        """
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            x_test = X_test[i:i+1]
            
            if self.distance_metric == 'euclidean':
                distances[i] = self._euclidean_distance(x_test, self.X_train)
            elif self.distance_metric == 'manhattan':
                distances[i] = self._manhattan_distance(x_test, self.X_train)
            elif self.distance_metric == 'minkowski':
                distances[i] = self._minkowski_distance(x_test, self.X_train)
            elif self.distance_metric == 'cosine':
                distances[i] = self._cosine_distance(x_test, self.X_train)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 测试数据，shape=(n_samples, n_features)
        
        返回:
            预测值，shape=(n_samples, 1)
        """
        # 计算距离
        distances = self._compute_distances(X)
        
        # 找到k个最近邻的索引
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # 获取k个最近邻的标签
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # 获取k个最近邻的距离
        k_nearest_distances = np.take_along_axis(distances, k_nearest_indices, axis=1)
        
        if self.task == 'classification':
            return self._predict_classification(k_nearest_labels, k_nearest_distances)
        else:
            return self._predict_regression(k_nearest_labels, k_nearest_distances)
    
    def _predict_classification(self, k_nearest_labels, k_nearest_distances):
        """
        分类预测：投票法
        """
        n_samples = k_nearest_labels.shape[0]
        predictions = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            labels = k_nearest_labels[i].ravel()
            distances = k_nearest_distances[i]
            
            if self.weights == 'uniform':
                # 简单投票
                predictions[i] = Counter(labels).most_common(1)[0][0]
            else:
                # 距离加权投票
                weights = 1 / (distances + 1e-8)
                weighted_votes = {}
                for label, weight in zip(labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                predictions[i] = max(weighted_votes, key=weighted_votes.get)
        
        return predictions
    
    def _predict_regression(self, k_nearest_labels, k_nearest_distances):
        """
        回归预测：平均值
        """
        if self.weights == 'uniform':
            # 简单平均
            return np.mean(k_nearest_labels, axis=1)
        else:
            # 距离加权平均
            weights = 1 / (k_nearest_distances + 1e-8)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            return np.sum(k_nearest_labels * weights[:, :, np.newaxis], axis=1)
    
    def score(self, X, y):
        """
        计算模型性能
        
        分类任务: 准确率
        回归任务: R²得分
        """
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            # R²得分
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def get_neighbors(self, x, return_distances=False):
        """
        获取单个样本的k个最近邻
        
        参数:
            x: 单个样本，shape=(n_features,) 或 (1, n_features)
            return_distances: 是否返回距离
        
        返回:
            neighbors: k个最近邻的索引
            distances: k个最近邻的距离（如果return_distances=True）
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        distances = self._compute_distances(x)[0]
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        if return_distances:
            k_nearest_distances = distances[k_nearest_indices]
            return k_nearest_indices, k_nearest_distances
        else:
            return k_nearest_indices


def generate_classification_data(n_samples=300, n_classes=3, n_features=2, 
                                 noise=0.3, random_state=42):
    """
    生成多分类数据
    """
    np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    for i in range(n_classes):
        # 每个类别在不同位置
        angle = 2 * np.pi * i / n_classes
        center = [3 * np.cos(angle), 3 * np.sin(angle)]
        
        X_class = np.random.randn(samples_per_class, n_features) * noise + center
        y_class = np.full(samples_per_class, i)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_regression_data(n_samples=200, noise=0.5, random_state=42):
    """
    生成回归数据
    """
    np.random.seed(random_state)
    
    X = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
    y = np.sin(X).ravel() + np.random.randn(n_samples) * noise
    
    return X, y


def plot_decision_boundary_2d(knn, X, y, title="KNN Decision Boundary"):
    """
    绘制2D决策边界
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制等高线
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                         edgecolors='black', s=50)
    plt.colorbar(scatter)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)


def example_classification():
    """
    示例1：KNN分类
    """
    print("=" * 70)
    print("示例1：KNN分类")
    print("=" * 70)
    
    # 生成分类数据
    X, y = generate_classification_data(n_samples=300, n_classes=3)
    
    # 划分训练集和测试集
    split_idx = 240
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 训练KNN
    print("\n训练KNN分类器 (k=5, euclidean距离)...")
    knn = KNN(k=5, task='classification', distance_metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)
    
    # 评估
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 示例预测
    test_sample = X_test[0]
    neighbors_idx, distances = knn.get_neighbors(test_sample, return_distances=True)
    pred = knn.predict(test_sample.reshape(1, -1))[0, 0]
    
    print(f"\n示例预测:")
    print(f"测试样本: {test_sample}")
    print(f"真实标签: {y_test[0]}")
    print(f"预测标签: {pred}")
    print(f"最近的{knn.k}个邻居的标签: {y_train[neighbors_idx]}")
    print(f"距离: {distances}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary_2d(knn, X, y, f"KNN (k={knn.k}) 决策边界")
    
    # 不同k值的比较
    k_values = [1, 5, 15, 30]
    accuracies = []
    
    for k in k_values:
        knn_temp = KNN(k=k, task='classification')
        knn_temp.fit(X_train, y_train)
        acc = knn_temp.score(X_test, y_test)
        accuracies.append(acc)
    
    plt.subplot(1, 3, 2)
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('k值')
    plt.ylabel('测试准确率')
    plt.title('不同k值的性能对比')
    plt.grid(True, alpha=0.3)
    
    # 最近邻可视化
    plt.subplot(1, 3, 3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
               cmap='viridis', alpha=0.3, s=50)
    plt.scatter(test_sample[0], test_sample[1], c='red', 
               marker='*', s=500, edgecolors='black', 
               linewidths=2, label='测试样本')
    plt.scatter(X_train[neighbors_idx, 0], X_train[neighbors_idx, 1], 
               c=y_train[neighbors_idx], cmap='viridis',
               s=200, edgecolors='red', linewidths=3, 
               label='最近邻')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('最近邻可视化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_classification.png', dpi=150)
    print("\n结果图已保存: knn_classification.png")
    plt.show()


def example_regression():
    """
    示例2：KNN回归
    """
    print("\n" + "=" * 70)
    print("示例2：KNN回归")
    print("=" * 70)
    
    # 生成回归数据
    X, y = generate_regression_data(n_samples=200, noise=0.3)
    
    # 划分数据集
    split_idx = 160
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 训练KNN回归
    print("\n训练KNN回归器 (k=5)...")
    knn_reg = KNN(k=5, task='regression', weights='distance')
    knn_reg.fit(X_train, y_train)
    
    # 评估
    train_r2 = knn_reg.score(X_train, y_train)
    test_r2 = knn_reg.score(X_test, y_test)
    
    print(f"\n训练集R²: {train_r2:.4f}")
    print(f"测试集R²: {test_r2:.4f}")
    
    # 预测
    X_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    y_pred = knn_reg.predict(X_plot)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # KNN回归拟合
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, c='blue', alpha=0.6, label='训练数据')
    plt.scatter(X_test, y_test, c='red', alpha=0.6, label='测试数据')
    plt.plot(X_plot, y_pred, 'g-', linewidth=2, label='KNN预测')
    plt.plot(X_plot, np.sin(X_plot), 'k--', linewidth=1, alpha=0.5, label='真实函数')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'KNN回归 (k={knn_reg.k}, 距离加权)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 不同k值对比
    plt.subplot(1, 3, 2)
    k_values = [1, 3, 5, 10, 20]
    
    for k in k_values:
        knn_temp = KNN(k=k, task='regression', weights='distance')
        knn_temp.fit(X_train, y_train)
        y_pred_temp = knn_temp.predict(X_plot)
        plt.plot(X_plot, y_pred_temp, linewidth=2, label=f'k={k}')
    
    plt.scatter(X_train, y_train, c='black', alpha=0.3, s=20)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('不同k值的回归曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R²得分对比
    plt.subplot(1, 3, 3)
    k_range = range(1, 31)
    r2_scores = []
    
    for k in k_range:
        knn_temp = KNN(k=k, task='regression', weights='distance')
        knn_temp.fit(X_train, y_train)
        r2 = knn_temp.score(X_test, y_test)
        r2_scores.append(r2)
    
    plt.plot(k_range, r2_scores, 'bo-', linewidth=2)
    plt.xlabel('k值')
    plt.ylabel('测试集R²')
    plt.title('不同k值的R²得分')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=max(r2_scores), color='r', linestyle='--', 
                label=f'最佳R²={max(r2_scores):.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('knn_regression.png', dpi=150)
    print("\n结果图已保存: knn_regression.png")
    plt.show()


def compare_distance_metrics():
    """
    示例3：不同距离度量对比
    """
    print("\n" + "=" * 70)
    print("示例3：不同距离度量对比")
    print("=" * 70)
    
    # 生成数据
    X, y = generate_classification_data(n_samples=300, n_classes=3)
    
    split_idx = 240
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 测试不同距离度量
    distance_metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']
    
    plt.figure(figsize=(15, 10))
    
    results = []
    
    for idx, metric in enumerate(distance_metrics):
        print(f"\n训练KNN with {metric} 距离...")
        knn = KNN(k=5, task='classification', distance_metric=metric)
        knn.fit(X_train, y_train)
        
        accuracy = knn.score(X_test, y_test)
        results.append((metric, accuracy))
        
        print(f"{metric} 距离准确率: {accuracy:.4f}")
        
        plt.subplot(2, 2, idx + 1)
        plot_decision_boundary_2d(knn, X, y, 
                                 f"{metric.capitalize()} (准确率={accuracy:.3f})")
    
    plt.tight_layout()
    plt.savefig('knn_distance_metrics.png', dpi=150)
    print("\n结果图已保存: knn_distance_metrics.png")
    plt.show()
    
    # 打印汇总
    print("\n" + "=" * 70)
    print(f"{'距离度量':<15} {'测试准确率':<15}")
    print("=" * 70)
    for metric, acc in results:
        print(f"{metric:<15} {acc:<15.4f}")
    print("=" * 70)


def compare_weighting_schemes():
    """
    示例4：对比uniform和distance加权
    """
    print("\n" + "=" * 70)
    print("示例4：对比投票方式")
    print("=" * 70)
    
    # 生成数据
    X, y = generate_classification_data(n_samples=300, n_classes=3)
    
    split_idx = 240
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 对比uniform和distance权重
    weights_list = ['uniform', 'distance']
    
    plt.figure(figsize=(12, 5))
    
    for idx, weights in enumerate(weights_list):
        print(f"\n训练KNN with {weights} 权重...")
        knn = KNN(k=5, task='classification', weights=weights)
        knn.fit(X_train, y_train)
        
        accuracy = knn.score(X_test, y_test)
        print(f"{weights} 权重准确率: {accuracy:.4f}")
        
        plt.subplot(1, 2, idx + 1)
        plot_decision_boundary_2d(knn, X, y, 
                                 f"{weights.capitalize()} 权重 (准确率={accuracy:.3f})")
    
    plt.tight_layout()
    plt.savefig('knn_weighting.png', dpi=150)
    print("\n结果图已保存: knn_weighting.png")
    plt.show()


def mathematical_explanation():
    """
    KNN数学原理说明
    """
    explanation = """
═══════════════════════════════════════════════════════════════════════
K近邻算法(KNN)数学原理
═══════════════════════════════════════════════════════════════════════

1. 核心思想
   
   基于"物以类聚"的思想：样本的类别由其最相似的k个邻居决定
   
   - 非参数方法：不需要训练，直接存储训练数据
   - 懒惰学习(Lazy Learning)：预测时才计算

2. 算法步骤

   分类任务:
   1) 计算测试样本与所有训练样本的距离
   2) 选择距离最近的k个训练样本
   3) 统计这k个样本中各类别出现的频率
   4) 返回频率最高的类别作为预测结果
   
   回归任务:
   1) 计算测试样本与所有训练样本的距离
   2) 选择距离最近的k个训练样本
   3) 返回这k个样本标签的平均值

3. 距离度量

   欧氏距离(Euclidean):
   d(x, y) = √(Σ(x_i - y_i)²)
   
   曼哈顿距离(Manhattan):
   d(x, y) = Σ|x_i - y_i|
   
   闵可夫斯基距离(Minkowski):
   d(x, y) = (Σ|x_i - y_i|^p)^(1/p)
   
   余弦距离(Cosine):
   d(x, y) = 1 - (x·y)/(||x||·||y||)

4. 投票方式

   简单投票(Uniform):
   class = argmax Σ I(y_i = c)
            c     i∈N_k
   
   距离加权投票(Distance):
   class = argmax Σ w_i·I(y_i = c)
            c     i∈N_k
   
   其中 w_i = 1/d_i（距离越近，权重越大）

5. 回归预测

   简单平均:
   ŷ = (1/k)·Σ y_i
              i∈N_k
   
   距离加权平均:
   ŷ = Σ w_i·y_i / Σ w_i
       i∈N_k        i∈N_k

6. k值的选择

   - k太小：对噪声敏感，容易过拟合
   - k太大：计算量大，可能欠拟合
   - 常用方法：交叉验证选择最优k
   - 经验规则：k = √n（n为训练样本数）

7. 优点
   
   - 简单直观，易于理解
   - 无需训练过程
   - 适用于多分类问题
   - 对异常值不敏感（当k较大时）

8. 缺点
   
   - 预测时计算量大（需要计算到所有训练样本的距离）
   - 需要大量内存存储训练数据
   - 对特征缩放敏感
   - 不适合高维数据（维度灾难）

9. 优化技巧
   
   - 特征标准化/归一化
   - 使用KD树或Ball树加速搜索
   - 降维（PCA等）
   - 特征选择
   
═══════════════════════════════════════════════════════════════════════
"""
    print(explanation)


if __name__ == "__main__":
    # 打印数学原理
    print("=" * 70)
    print("K近邻算法(KNN) - NumPy实现")
    print("=" * 70)
    mathematical_explanation()
    
    # 运行示例
    example_classification()
    example_regression()
    compare_distance_metrics()
    compare_weighting_schemes()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
