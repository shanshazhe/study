"""
决策树实现示例
包含：
1. NumPy从零实现决策树（分类）
2. Scikit-learn决策树
3. XGBoost梯度提升树

决策树核心思想：通过递归地选择最优特征进行分裂，构建树结构
"""

import numpy as np
from collections import Counter


# =============================================================================
# 第一部分：NumPy从零实现决策树
# =============================================================================

class DecisionTreeNode:
    """决策树节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        """
        参数:
            feature_idx: 分裂特征的索引
            threshold: 分裂阈值
            left: 左子树（特征值 <= threshold）
            right: 右子树（特征值 > threshold）
            value: 叶节点的预测值（类别）
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierNumpy:
    """
    使用NumPy实现的决策树分类器
    
    分裂准则支持：
    - 信息增益 (Information Gain) - 基于熵
    - 基尼不纯度 (Gini Impurity)
    
    数学原理：
    
    1. 熵 (Entropy):
       H(S) = -Σ p_i * log2(p_i)
       其中 p_i 是类别 i 的概率
    
    2. 信息增益 (Information Gain):
       IG(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)
       选择信息增益最大的特征进行分裂
    
    3. 基尼不纯度 (Gini Impurity):
       Gini(S) = 1 - Σ p_i²
       选择基尼不纯度最小的分裂点
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        """
        参数:
            max_depth: 树的最大深度
            min_samples_split: 节点分裂所需的最小样本数
            criterion: 分裂准则 ('gini' 或 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None
    
    def _entropy(self, y):
        """
        计算熵
        H(S) = -Σ p_i * log2(p_i)
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        # 过滤掉概率为0的项（避免log(0)）
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))
    
    def _gini(self, y):
        """
        计算基尼不纯度
        Gini(S) = 1 - Σ p_i²
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps ** 2)
    
    def _impurity(self, y):
        """根据criterion选择不纯度计算方法"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        return self._gini(y)
    
    def _information_gain(self, y, left_idxs, right_idxs):
        """
        计算信息增益
        IG = H(parent) - weighted_avg(H(children))
        """
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        
        # 父节点不纯度
        parent_impurity = self._impurity(y)
        
        # 子节点加权不纯度
        left_impurity = self._impurity(y[left_idxs])
        right_impurity = self._impurity(y[right_idxs])
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """
        找到最佳分裂点
        遍历所有特征和所有可能的阈值，选择信息增益最大的分裂
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # 根据阈值分裂
                left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                # 计算信息增益
                gain = self._information_gain(y, left_idxs, right_idxs)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        # 1. 达到最大深度
        # 2. 样本数少于最小分裂数
        # 3. 所有样本属于同一类别
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            # 创建叶节点，预测值为多数类
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # 找最佳分裂点
        best_feature_idx, best_threshold, best_gain = self._best_split(X, y)
        
        # 如果无法获得信息增益，创建叶节点
        if best_gain <= 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # 分裂数据
        left_idxs = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_idxs = np.where(X[:, best_feature_idx] > best_threshold)[0]
        
        # 递归构建子树
        left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """训练决策树"""
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """对单个样本进行预测"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X):
        """预测"""
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """打印决策树结构"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf():
            print(f"{indent}{prefix}: 类别 = {node.value}")
        else:
            print(f"{indent}{prefix}: 特征[{node.feature_idx}] <= {node.threshold:.4f}")
            self.print_tree(node.left, depth + 1, "左")
            self.print_tree(node.right, depth + 1, "右")


def generate_classification_data(n_samples=200, n_features=4, n_classes=3, random_state=42):
    """生成分类数据"""
    np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # 每个类别有不同的中心
        center = np.random.randn(n_features) * 3
        X_class = center + np.random.randn(samples_per_class, n_features)
        X.append(X_class)
        y.extend([class_idx] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # 打乱数据
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def example_numpy_decision_tree():
    """
    示例1：NumPy实现的决策树
    """
    print("=" * 70)
    print("示例1：NumPy从零实现决策树")
    print("=" * 70)
    
    # 生成数据
    X, y = generate_classification_data(n_samples=300, n_features=4, n_classes=3)
    
    # 划分训练集和测试集
    split_idx = 240
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n数据集信息:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  类别数: {len(np.unique(y))}")
    
    # 训练决策树（使用基尼不纯度）
    print("\n--- 使用基尼不纯度 ---")
    tree_gini = DecisionTreeClassifierNumpy(max_depth=5, min_samples_split=5, criterion='gini')
    tree_gini.fit(X_train, y_train)
    
    train_acc = tree_gini.score(X_train, y_train)
    test_acc = tree_gini.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 训练决策树（使用信息熵）
    print("\n--- 使用信息熵 ---")
    tree_entropy = DecisionTreeClassifierNumpy(max_depth=5, min_samples_split=5, criterion='entropy')
    tree_entropy.fit(X_train, y_train)
    
    train_acc = tree_entropy.score(X_train, y_train)
    test_acc = tree_entropy.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 打印树结构
    print("\n决策树结构（基尼不纯度，max_depth=3）:")
    tree_small = DecisionTreeClassifierNumpy(max_depth=3, criterion='gini')
    tree_small.fit(X_train, y_train)
    tree_small.print_tree()
    
    return tree_gini


# =============================================================================
# 第二部分：Scikit-learn决策树
# =============================================================================

def example_sklearn_decision_tree():
    """
    示例2：Scikit-learn决策树
    """
    print("\n" + "=" * 70)
    print("示例2：Scikit-learn决策树")
    print("=" * 70)
    
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.datasets import load_iris, load_wine
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("请安装scikit-learn: pip install scikit-learn")
        return
    
    # 使用Iris数据集
    print("\n--- Iris数据集 ---")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"特征名称: {iris.feature_names}")
    print(f"类别名称: {iris.target_names.tolist()}")
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 创建决策树
    clf = DecisionTreeClassifier(
        criterion='gini',      # 分裂准则：'gini' 或 'entropy'
        max_depth=4,           # 最大深度
        min_samples_split=5,   # 节点分裂最小样本数
        min_samples_leaf=2,    # 叶节点最小样本数
        random_state=42
    )
    
    # 训练
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 评估
    print(f"\n训练集准确率: {clf.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {clf.score(X_test, y_test):.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"5折交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 特征重要性
    print("特征重要性:")
    for name, importance in zip(iris.feature_names, clf.feature_importances_):
        print(f"  {name}: {importance:.4f}")
    
    # 打印决策树规则
    print("\n决策树规则:")
    tree_rules = export_text(clf, feature_names=iris.feature_names)
    print(tree_rules)
    
    # 使用Wine数据集
    print("\n--- Wine数据集 ---")
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 比较不同深度
    print("\n不同最大深度的性能对比:")
    print(f"{'深度':<8} {'训练准确率':<12} {'测试准确率':<12} {'过拟合程度':<12}")
    print("-" * 50)
    
    for max_depth in [2, 3, 5, 10, None]:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        overfit = train_acc - test_acc
        depth_str = str(max_depth) if max_depth else "无限"
        print(f"{depth_str:<8} {train_acc:<12.4f} {test_acc:<12.4f} {overfit:<12.4f}")
    
    return clf


# =============================================================================
# 第三部分：XGBoost梯度提升树
# =============================================================================

def example_xgboost():
    """
    示例3：XGBoost梯度提升树
    
    XGBoost核心思想：
    1. 梯度提升：迭代地添加新树来纠正之前模型的残差
    2. 正则化：在目标函数中加入树的复杂度惩罚
    3. 二阶泰勒展开：使用二阶导数信息加速优化
    
    目标函数：
    Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
    
    其中：
    - L: 损失函数
    - Ω(f): 正则化项 = γT + (1/2)λΣw_j²
    - T: 叶子节点数
    - w_j: 叶子节点权重
    """
    print("\n" + "=" * 70)
    print("示例3：XGBoost梯度提升树")
    print("=" * 70)
    
    try:
        import xgboost as xgb
        from sklearn.datasets import load_breast_cancer, make_classification
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, roc_auc_score
    except ImportError:
        print("请安装xgboost: pip install xgboost")
        return
    
    # 使用乳腺癌数据集
    print("\n--- 乳腺癌数据集（二分类）---")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"特征数: {X.shape[1]}")
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"正样本比例: {y.mean():.2%}")
    
    # 创建XGBoost分类器
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,        # 树的数量
        max_depth=4,             # 每棵树的最大深度
        learning_rate=0.1,       # 学习率（步长收缩）
        subsample=0.8,           # 每棵树使用的样本比例
        colsample_bytree=0.8,    # 每棵树使用的特征比例
        reg_alpha=0.1,           # L1正则化
        reg_lambda=1.0,          # L2正则化
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    
    # 训练（带早停）
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
    
    # 评估
    print(f"\n训练集准确率: {xgb_clf.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {xgb_clf.score(X_test, y_test):.4f}")
    print(f"测试集AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))
    
    # 特征重要性
    print("\nTop 10 重要特征:")
    feature_importance = xgb_clf.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {data.feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # 多分类示例
    print("\n--- 多分类示例 ---")
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=4, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )
    
    xgb_multi = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        verbosity=0
    )
    
    xgb_multi.fit(X_train, y_train)
    
    print(f"训练集准确率: {xgb_multi.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {xgb_multi.score(X_test, y_test):.4f}")
    
    # 超参数对比
    print("\n--- XGBoost超参数影响 ---")
    print(f"{'参数设置':<30} {'测试准确率':<12}")
    print("-" * 45)
    
    # 不同学习率
    for lr in [0.01, 0.1, 0.3]:
        clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=lr, max_depth=4,
            random_state=42, verbosity=0
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"learning_rate={lr:<20} {acc:.4f}")
    
    # 不同深度
    for depth in [2, 4, 6, 8]:
        clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=depth,
            random_state=42, verbosity=0
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"max_depth={depth:<23} {acc:.4f}")
    
    return xgb_clf


# =============================================================================
# 第四部分：三种方法对比
# =============================================================================

def compare_all_methods():
    """
    对比NumPy实现、Scikit-learn和XGBoost
    """
    print("\n" + "=" * 70)
    print("三种实现方法对比")
    print("=" * 70)
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        import time
    except ImportError:
        print("请安装依赖: pip install scikit-learn xgboost")
        return
    
    # 生成数据
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    results = []
    
    # 1. NumPy实现
    print("\n训练中...")
    start = time.time()
    tree_numpy = DecisionTreeClassifierNumpy(max_depth=10, criterion='gini')
    tree_numpy.fit(X_train, y_train)
    numpy_time = time.time() - start
    numpy_train_acc = tree_numpy.score(X_train, y_train)
    numpy_test_acc = tree_numpy.score(X_test, y_test)
    results.append(('NumPy决策树', numpy_train_acc, numpy_test_acc, numpy_time))
    
    # 2. Scikit-learn
    start = time.time()
    tree_sklearn = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree_sklearn.fit(X_train, y_train)
    sklearn_time = time.time() - start
    sklearn_train_acc = tree_sklearn.score(X_train, y_train)
    sklearn_test_acc = tree_sklearn.score(X_test, y_test)
    results.append(('Sklearn决策树', sklearn_train_acc, sklearn_test_acc, sklearn_time))
    
    # 3. XGBoost
    start = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    xgb_clf.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_train_acc = xgb_clf.score(X_train, y_train)
    xgb_test_acc = xgb_clf.score(X_test, y_test)
    results.append(('XGBoost', xgb_train_acc, xgb_test_acc, xgb_time))
    
    # 打印结果
    print(f"\n{'方法':<15} {'训练准确率':<12} {'测试准确率':<12} {'训练时间(秒)':<12}")
    print("-" * 55)
    for name, train_acc, test_acc, train_time in results:
        print(f"{name:<15} {train_acc:<12.4f} {test_acc:<12.4f} {train_time:<12.4f}")
    
    print("\n结论:")
    print("  - NumPy实现：教学用途，速度较慢")
    print("  - Sklearn：生产级别，单棵树，速度快")
    print("  - XGBoost：集成多棵树，通常准确率最高")


def mathematical_explanation():
    """
    决策树数学原理说明
    """
    explanation = """
═══════════════════════════════════════════════════════════════════════
决策树数学原理
═══════════════════════════════════════════════════════════════════════

1. 信息熵 (Entropy)
   
   H(S) = -Σ p_i * log₂(p_i)
   
   - 衡量数据集的不确定性
   - 熵越大，不确定性越高
   - 纯净数据集（只有一个类别）熵为0

2. 信息增益 (Information Gain)
   
   IG(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)
   
   - 选择信息增益最大的特征进行分裂
   - ID3算法使用信息增益

3. 基尼不纯度 (Gini Impurity)
   
   Gini(S) = 1 - Σ p_i²
   
   - CART算法使用基尼不纯度
   - 计算比熵更快（无需对数运算）
   - sklearn默认使用基尼不纯度

4. 增益率 (Gain Ratio)
   
   GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
   
   - C4.5算法使用增益率
   - 解决信息增益偏向多值特征的问题

5. 决策树构建过程
   
   function BuildTree(S, features):
       if 停止条件满足:
           return 叶节点(多数类)
       
       best_feature = argmax(InformationGain)
       tree = 以best_feature创建节点
       
       for value in best_feature.values:
           S_v = {s ∈ S | s.best_feature = value}
           tree.add_child(BuildTree(S_v, features - {best_feature}))
       
       return tree

6. 剪枝策略
   
   预剪枝 (Pre-pruning):
   - 限制最大深度 (max_depth)
   - 限制最小分裂样本数 (min_samples_split)
   - 限制最小叶节点样本数 (min_samples_leaf)
   
   后剪枝 (Post-pruning):
   - 成本复杂度剪枝 (Cost Complexity Pruning)
   - 悲观剪枝 (Pessimistic Pruning)

7. XGBoost目标函数
   
   Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
   
   正则化项: Ω(f) = γT + (1/2)λΣw_j²
   
   - γ: 叶子节点数量惩罚
   - λ: 权重L2正则化
   - T: 叶子节点数
   - w_j: 第j个叶子的权重

8. XGBoost分裂增益
   
   Gain = (1/2) * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
   
   - G: 一阶导数之和
   - H: 二阶导数之和
   - 选择增益最大的分裂点

═══════════════════════════════════════════════════════════════════════
"""
    print(explanation)


if __name__ == "__main__":
    # 打印数学原理
    mathematical_explanation()
    
    # 运行所有示例
    example_numpy_decision_tree()
    example_sklearn_decision_tree()
    example_xgboost()
    compare_all_methods()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
