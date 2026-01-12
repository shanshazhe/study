import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PCA:
    """
    主成分分析(PCA)的NumPy实现
    支持协方差矩阵和SVD两种方法
    """
    def __init__(self, n_components=2, method='covariance'):
        """
        参数:
            n_components: 保留的主成分数量
            method: 'covariance' 或 'svd'
        """
        self.n_components = n_components
        self.method = method
        self.components_ = None  # 主成分（特征向量）
        self.explained_variance_ = None  # 解释方差
        self.explained_variance_ratio_ = None  # 方差解释率
        self.mean_ = None  # 数据均值
        self.singular_values_ = None  # 奇异值（仅SVD方法）
    
    def fit(self, X):
        """
        训练PCA模型
        
        参数:
            X: 输入数据，shape=(n_samples, n_features)
        """
        if self.method == 'covariance':
            self._fit_covariance(X)
        elif self.method == 'svd':
            self._fit_svd(X)
        else:
            raise ValueError("method must be 'covariance' or 'svd'")
        
        return self
    
    def _fit_covariance(self, X):
        """
        方法1：使用协方差矩阵的特征值分解
        
        步骤:
            1. 中心化数据：X_centered = X - mean(X)
            2. 计算协方差矩阵：C = 1/(n-1) * X_centered^T * X_centered
            3. 特征值分解：C = V * Λ * V^T
            4. 选择前k个最大特征值对应的特征向量
        """
        n_samples, n_features = X.shape
        
        # 1. 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. 计算协方差矩阵
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # 3. 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. 选择前n_components个主成分
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # 6. 计算方差解释率
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
    
    def _fit_svd(self, X):
        """
        方法2：使用奇异值分解(SVD)
        
        步骤:
            1. 中心化数据：X_centered = X - mean(X)
            2. SVD分解：X_centered = U * Σ * V^T
            3. 主成分即为V的列向量
            4. 方差 = (奇异值²) / (n-1)
        
        优点:
            - 数值稳定性更好
            - 不需要显式计算协方差矩阵
            - 计算效率更高（特别是当n_features很大时）
        """
        n_samples, n_features = X.shape
        
        # 1. 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. SVD分解
        # U: (n_samples, n_samples), 左奇异向量
        # S: (min(n_samples, n_features),), 奇异值
        # Vt: (n_features, n_features), 右奇异向量的转置
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. 主成分是V的前n_components行（即Vt的前n_components行）
        self.components_ = Vt[:self.n_components, :]
        
        # 4. 计算解释方差
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n_samples - 1)
        
        # 5. 计算方差解释率
        total_variance = np.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
    
    def transform(self, X):
        """
        将数据转换到主成分空间
        
        参数:
            X: 输入数据，shape=(n_samples, n_features)
        
        返回:
            X_transformed: 转换后的数据，shape=(n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """
        训练并转换数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        从主成分空间还原到原始空间
        
        参数:
            X_transformed: 主成分空间的数据，shape=(n_samples, n_components)
        
        返回:
            X_original: 还原后的数据，shape=(n_samples, n_features)
        """
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def get_cumulative_variance_ratio(self):
        """
        获取累积方差解释率
        """
        return np.cumsum(self.explained_variance_ratio_)


def generate_correlated_data(n_samples=300, random_state=42):
    """
    生成具有相关性的3D数据
    """
    np.random.seed(random_state)
    
    # 生成高度相关的3维数据
    mean = [0, 0, 0]
    cov = [[3.0, 2.5, 2.0],
           [2.5, 3.0, 2.5],
           [2.0, 2.5, 3.0]]
    
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    return X


def example_2d_to_1d():
    """
    示例1：2D数据降维到1D
    """
    print("=" * 70)
    print("示例1：2D数据降维到1D")
    print("=" * 70)
    
    # 生成2D数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    X[:, 1] = X[:, 0] * 2 + np.random.randn(200) * 0.5  # y与x相关
    
    print(f"\n原始数据形状: {X.shape}")
    
    # 方法1：协方差矩阵
    print("\n--- 方法1：协方差矩阵特征值分解 ---")
    pca_cov = PCA(n_components=1, method='covariance')
    X_pca_cov = pca_cov.fit_transform(X)
    
    print(f"主成分方向: {pca_cov.components_}")
    print(f"解释方差: {pca_cov.explained_variance_}")
    print(f"方差解释率: {pca_cov.explained_variance_ratio_}")
    print(f"降维后数据形状: {X_pca_cov.shape}")
    
    # 方法2：SVD
    print("\n--- 方法2：奇异值分解(SVD) ---")
    pca_svd = PCA(n_components=1, method='svd')
    X_pca_svd = pca_svd.fit_transform(X)
    
    print(f"主成分方向: {pca_svd.components_}")
    print(f"奇异值: {pca_svd.singular_values_}")
    print(f"解释方差: {pca_svd.explained_variance_}")
    print(f"方差解释率: {pca_svd.explained_variance_ratio_}")
    print(f"降维后数据形状: {X_pca_svd.shape}")
    
    # 重构数据
    X_reconstructed_cov = pca_cov.inverse_transform(X_pca_cov)
    X_reconstructed_svd = pca_svd.inverse_transform(X_pca_svd)
    
    reconstruction_error_cov = np.mean((X - X_reconstructed_cov) ** 2)
    reconstruction_error_svd = np.mean((X - X_reconstructed_svd) ** 2)
    
    print(f"\n重构误差 (协方差方法): {reconstruction_error_cov:.6f}")
    print(f"重构误差 (SVD方法): {reconstruction_error_svd:.6f}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 原始数据和主成分方向
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    
    # 绘制主成分方向
    mean = np.mean(X, axis=0)
    pc1 = pca_cov.components_[0]
    plt.arrow(mean[0], mean[1], pc1[0]*3, pc1[1]*3, 
              head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('原始数据与第一主成分')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 降维后的数据
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca_cov, np.zeros_like(X_pca_cov), alpha=0.5)
    plt.xlabel('主成分1')
    plt.ylabel('')
    plt.title('降维到1D')
    plt.grid(True, alpha=0.3)
    
    # 重构数据对比
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='原始数据')
    plt.scatter(X_reconstructed_cov[:, 0], X_reconstructed_cov[:, 1], 
                alpha=0.3, label='重构数据')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('原始数据 vs 重构数据')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pca_2d_to_1d.png', dpi=150)
    print("\n结果图已保存: pca_2d_to_1d.png")
    plt.show()


def example_3d_to_2d():
    """
    示例2：3D数据降维到2D
    """
    print("\n" + "=" * 70)
    print("示例2：3D数据降维到2D")
    print("=" * 70)
    
    # 生成3D相关数据
    X = generate_correlated_data(n_samples=300)
    
    print(f"\n原始数据形状: {X.shape}")
    print(f"原始数据方差: {np.var(X, axis=0)}")
    
    # 使用SVD方法
    pca = PCA(n_components=2, method='svd')
    X_pca = pca.fit_transform(X)
    
    print(f"\n主成分分析结果:")
    print(f"主成分形状: {pca.components_.shape}")
    print(f"各主成分方向:\n{pca.components_}")
    print(f"解释方差: {pca.explained_variance_}")
    print(f"方差解释率: {pca.explained_variance_ratio_}")
    print(f"累积方差解释率: {pca.get_cumulative_variance_ratio()}")
    print(f"降维后数据形状: {X_pca.shape}")
    
    # 可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 3D原始数据
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('原始3D数据')
    
    # 降维到2D
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    ax2.set_xlabel('主成分1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
    ax2.set_ylabel('主成分2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
    ax2.set_title('降维到2D')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 方差解释率
    ax3 = fig.add_subplot(133)
    components = ['PC1', 'PC2']
    ax3.bar(components, pca.explained_variance_ratio_, alpha=0.7)
    ax3.set_ylabel('方差解释率')
    ax3.set_title('各主成分的方差解释率')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(pca.explained_variance_ratio_):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pca_3d_to_2d.png', dpi=150)
    print("\n结果图已保存: pca_3d_to_2d.png")
    plt.show()


def compare_methods():
    """
    示例3：比较协方差和SVD两种方法
    """
    print("\n" + "=" * 70)
    print("示例3：协方差 vs SVD 方法对比")
    print("=" * 70)
    
    # 生成高维数据
    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.randn(n_samples, n_features)
    
    # 添加相关性
    for i in range(1, n_features):
        X[:, i] += X[:, 0] * (n_features - i) / n_features
    
    print(f"\n数据形状: {X.shape}")
    
    # 测试不同数量的主成分
    n_components_list = [2, 5, 8]
    
    print("\n" + "-" * 70)
    print(f"{'主成分数':<10} {'方法':<15} {'计算时间(ms)':<15} {'累积方差解释率':<20}")
    print("-" * 70)
    
    import time
    
    for n_comp in n_components_list:
        # 协方差方法
        start = time.time()
        pca_cov = PCA(n_components=n_comp, method='covariance')
        pca_cov.fit(X)
        time_cov = (time.time() - start) * 1000
        cum_var_cov = pca_cov.get_cumulative_variance_ratio()[-1]
        
        # SVD方法
        start = time.time()
        pca_svd = PCA(n_components=n_comp, method='svd')
        pca_svd.fit(X)
        time_svd = (time.time() - start) * 1000
        cum_var_svd = pca_svd.get_cumulative_variance_ratio()[-1]
        
        print(f"{n_comp:<10} {'Covariance':<15} {time_cov:<15.4f} {cum_var_cov:<20.4f}")
        print(f"{n_comp:<10} {'SVD':<15} {time_svd:<15.4f} {cum_var_svd:<20.4f}")
        print()
    
    # 可视化所有主成分的方差解释率
    pca_full = PCA(n_components=n_features, method='svd')
    pca_full.fit(X)
    
    plt.figure(figsize=(12, 5))
    
    # 方差解释率
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_features+1), pca_full.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('主成分编号')
    plt.ylabel('方差解释率')
    plt.title('各主成分的方差解释率')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 累积方差解释率
    plt.subplot(1, 2, 2)
    cumsum = pca_full.get_cumulative_variance_ratio()
    plt.plot(range(1, n_features+1), cumsum, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    plt.xlabel('主成分数量')
    plt.ylabel('累积方差解释率')
    plt.title('累积方差解释率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 找到达到95%方差的主成分数
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    plt.axvline(x=n_components_95, color='g', linestyle='--', 
                label=f'95%需要{n_components_95}个主成分')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png', dpi=150)
    print(f"需要 {n_components_95} 个主成分来解释95%的方差")
    print("\n结果图已保存: pca_comparison.png")
    plt.show()


def mathematical_explanation():
    """
    打印PCA的数学原理说明
    """
    print("\n" + "=" * 70)
    print("PCA数学原理")
    print("=" * 70)
    
    explanation = """
主成分分析(PCA)是一种无监督降维算法，通过线性变换将数据投影到方差最大的方向。

═══════════════════════════════════════════════════════════════════════

方法1：协方差矩阵的特征值分解

步骤:
  1. 数据中心化: X_centered = X - mean(X)
  
  2. 计算协方差矩阵:
     C = (1/(n-1)) * X_centered^T * X_centered
     
  3. 特征值分解:
     C * v = λ * v
     其中 v 是特征向量（主成分方向），λ 是特征值（方差大小）
     
  4. 按特征值降序排列，选择前k个特征向量作为主成分
  
  5. 投影: X_new = X_centered * V_k
     其中 V_k 是前k个特征向量组成的矩阵

优点: 直观易懂，有明确的方差解释
缺点: 需要显式计算协方差矩阵，当特征数很大时计算量大

═══════════════════════════════════════════════════════════════════════

方法2：奇异值分解(SVD)

步骤:
  1. 数据中心化: X_centered = X - mean(X)
  
  2. SVD分解:
     X_centered = U * Σ * V^T
     
     其中:
     - U: (n_samples × n_samples) 左奇异向量矩阵
     - Σ: (min(n,m) × min(n,m)) 奇异值对角矩阵
     - V^T: (n_features × n_features) 右奇异向量矩阵的转置
     
  3. 主成分即为V的列向量（或V^T的行向量）
  
  4. 方差与奇异值的关系:
     explained_variance = σ² / (n-1)
     
  5. 投影: X_new = U * Σ[:k]
     或者: X_new = X_centered * V[:, :k]

优点:
  - 数值稳定性更好
  - 不需要显式计算协方差矩阵
  - 适用于高维数据
  - 现代机器学习库的标准方法

关系: 当对协方差矩阵做SVD分解时，V矩阵等同于协方差矩阵的特征向量

═══════════════════════════════════════════════════════════════════════

关键概念:

1. 主成分: 数据方差最大的方向（正交）
2. 解释方差: 每个主成分捕获的数据方差
3. 方差解释率: 该主成分方差占总方差的比例
4. 累积方差解释率: 前k个主成分的方差解释率之和

应用场景:
  - 降维和数据可视化
  - 特征提取和特征工程
  - 噪声过滤
  - 数据压缩
  - 异常检测
"""
    print(explanation)


if __name__ == "__main__":
    # 打印数学原理
    mathematical_explanation()
    
    # 运行示例
    example_2d_to_1d()
    example_3d_to_2d()
    compare_methods()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
