import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 残差块(Residual Block)
# =============================================================================

class BasicBlock(nn.Module):
    """
    基础残差块（用于ResNet-18/34）
    
    结构:
        x -> [Conv-BN-ReLU] -> [Conv-BN] -> (+) -> ReLU -> output
        |                                    ^
        |____________________________________|
                   (shortcut)
    """
    expansion = 1  # 输出通道数相对于输入通道数的扩展倍数
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长（用于下采样）
            downsample: 下采样层（当输入输出维度不匹配时使用）
        """
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut连接（当维度不匹配时需要调整）
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # shortcut路径
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    瓶颈残差块（用于ResNet-50/101/152）
    
    结构:
        x -> [1x1 Conv-BN-ReLU] -> [3x3 Conv-BN-ReLU] -> [1x1 Conv-BN] -> (+) -> ReLU -> output
        |                                                                   ^
        |___________________________________________________________________|
                                    (shortcut)
    
    使用1x1卷积降低计算量
    """
    expansion = 4  # 输出通道数是中间通道数的4倍
    
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 3x3卷积
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # shortcut路径
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = F.relu(out)
        
        return out


# =============================================================================
# ResNet网络
# =============================================================================

class ResNet(nn.Module):
    """
    残差神经网络(ResNet)
    
    支持ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    """
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        """
        参数:
            block: 残差块类型 (BasicBlock 或 Bottleneck)
            layers: 每个stage的残差块数量，例如[2, 2, 2, 2]表示ResNet-18
            num_classes: 分类类别数
            in_channels: 输入通道数
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4个残差stage
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建残差stage
        
        参数:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 该stage中的残差块数量
            stride: 第一个残差块的步长
        """
        downsample = None
        
        # 如果维度不匹配，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # 第一个残差块（可能需要下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # 更新输入通道数
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 4个残差stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


# =============================================================================
# 不同深度的ResNet模型
# =============================================================================

def ResNet18(num_classes=10, in_channels=3):
    """ResNet-18: [2, 2, 2, 2]"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes=10, in_channels=3):
    """ResNet-34: [3, 4, 6, 3]"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes=10, in_channels=3):
    """ResNet-50: [3, 4, 6, 3] with Bottleneck"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def ResNet101(num_classes=10, in_channels=3):
    """ResNet-101: [3, 4, 23, 3] with Bottleneck"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def ResNet152(num_classes=10, in_channels=3):
    """ResNet-152: [3, 8, 36, 3] with Bottleneck"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


# =============================================================================
# 简化版ResNet（适用于CIFAR-10等小图像）
# =============================================================================

class ResNetCIFAR(nn.Module):
    """
    适用于CIFAR-10的简化ResNet（32x32输入）
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        
        self.in_channels = 16
        
        # 初始卷积（适合小图像，不使用7x7和maxpool）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 3个残差stage
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def ResNet20_CIFAR(num_classes=10):
    """ResNet-20 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes)


def ResNet32_CIFAR(num_classes=10):
    """ResNet-32 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes)


def ResNet56_CIFAR(num_classes=10):
    """ResNet-56 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes)


# =============================================================================
# 训练和评估函数
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_model(model, train_loader, test_loader, num_epochs=10, 
                learning_rate=0.1, device='cpu'):
    """完整训练流程"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=[num_epochs//2, 3*num_epochs//4], 
                                               gamma=0.1)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    print("开始训练...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("=" * 70)
    print("训练完成!")
    
    return train_losses, train_accs, test_losses, test_accs


# =============================================================================
# 示例：在CIFAR-10上训练ResNet
# =============================================================================

def example_cifar10():
    """
    在CIFAR-10数据集上训练ResNet
    """
    print("=" * 70)
    print("ResNet在CIFAR-10上的训练示例")
    print("=" * 70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    print("\n加载CIFAR-10数据集...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"类别: {classes}")
    
    # 创建模型
    print("\n创建ResNet-20模型...")
    model = ResNet20_CIFAR(num_classes=10).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 训练模型
    train_losses, train_accs, test_losses, test_accs = train_model(
        model, trainloader, testloader, 
        num_epochs=20, learning_rate=0.1, device=device
    )
    
    # 可视化训练过程
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet20_cifar10.pth')
    print("\n模型已保存: resnet20_cifar10.pth")
    
    return model, (train_losses, train_accs, test_losses, test_accs)


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """可视化训练历史"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和测试损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练和测试准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最终性能
    plt.subplot(1, 3, 3)
    metrics = ['Train Acc', 'Test Acc']
    values = [train_accs[-1], test_accs[-1]]
    colors = ['blue', 'orange']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Accuracy (%)')
    plt.title('最终准确率')
    plt.ylim([0, 100])
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('resnet_training.png', dpi=150)
    print("\n训练曲线已保存: resnet_training.png")
    plt.show()


def compare_architectures():
    """
    比较不同ResNet架构
    """
    print("\n" + "=" * 70)
    print("ResNet架构对比")
    print("=" * 70)
    
    models = {
        'ResNet-18': ResNet18(num_classes=10),
        'ResNet-34': ResNet34(num_classes=10),
        'ResNet-50': ResNet50(num_classes=10),
        'ResNet-20 (CIFAR)': ResNet20_CIFAR(num_classes=10),
        'ResNet-32 (CIFAR)': ResNet32_CIFAR(num_classes=10),
    }
    
    print(f"\n{'模型':<20} {'参数量':<15} {'输入大小':<15}")
    print("-" * 70)
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        
        # 测试输入
        if 'CIFAR' in name:
            test_input = torch.randn(1, 3, 32, 32)
        else:
            test_input = torch.randn(1, 3, 224, 224)
        
        print(f"{name:<20} {total_params:>12,} 参数  {str(test_input.shape[2:]):<15}")


def mathematical_explanation():
    """
    ResNet数学原理说明
    """
    explanation = """
═══════════════════════════════════════════════════════════════════════
残差神经网络(ResNet)数学原理
═══════════════════════════════════════════════════════════════════════

1. 核心思想：残差学习

   传统网络: H(x) = F(x)
   残差网络: H(x) = F(x) + x
   
   其中:
   - H(x): 期望的映射
   - F(x): 残差函数（需要学习的部分）
   - x: 输入（恒等映射）

2. 动机：解决深度网络的退化问题

   问题: 随着网络深度增加，训练误差反而上升（不是过拟合）
   
   原因: 深层网络难以学习恒等映射
   
   解决: 如果恒等映射是最优的，学习F(x)=0比学习H(x)=x更容易

3. 残差块数学表达式

   基础残差块:
   
   y = F(x, {W_i}) + x
   
   其中 F(x, {W_i}) = W_2 * σ(W_1 * x)
   
   完整表达式:
   
   y = F(x, {W_i}) + W_s * x
   
   当维度匹配时，W_s = I（恒等矩阵）
   当维度不匹配时，W_s用于调整维度

4. 前向传播

   l层的输出:
   
   x_l = x_{l-1} + F(x_{l-1}, W_l)
   
   展开后:
   
   x_L = x_l + Σ F(x_i, W_i)  (从l到L-1)
         i=l

5. 反向传播

   损失函数对x_l的梯度:
   
   ∂loss/∂x_l = ∂loss/∂x_L * ∂x_L/∂x_l
              = ∂loss/∂x_L * (1 + ∂/∂x_l Σ F(x_i, W_i))
   
   关键: 梯度中包含恒等项"1"，保证梯度能够直接传播
   
   优点: 避免梯度消失，即使F的梯度很小，总梯度仍不为0

6. 瓶颈设计(Bottleneck Block)

   结构: 1×1 Conv → 3×3 Conv → 1×1 Conv
   
   参数量对比:
   - 两个3×3 Conv (256通道): 2 × (3×3×256×256) ≈ 1.18M
   - Bottleneck (64→64→256): 1×1×256×64 + 3×3×64×64 + 1×1×64×256 ≈ 70K
   
   减少约94%的参数量！

7. 批归一化(Batch Normalization)

   作用:
   - 加速训练
   - 允许使用更大的学习率
   - 减少对初始化的依赖
   - 起到正则化作用

8. ResNet变体

   ResNet-18:  [2, 2, 2, 2] BasicBlock  → 18层
   ResNet-34:  [3, 4, 6, 3] BasicBlock  → 34层
   ResNet-50:  [3, 4, 6, 3] Bottleneck → 50层
   ResNet-101: [3, 4, 23, 3] Bottleneck → 101层
   ResNet-152: [3, 8, 36, 3] Bottleneck → 152层

9. 关键优势

   ✓ 可以训练非常深的网络（>100层）
   ✓ 解决梯度消失问题
   ✓ 性能随深度增加而提升
   ✓ 易于优化
   ✓ 计算效率高（瓶颈设计）

10. 应用领域

    - 图像分类
    - 目标检测
    - 图像分割
    - 人脸识别
    - 迁移学习的首选backbone

═══════════════════════════════════════════════════════════════════════
"""
    print(explanation)


if __name__ == "__main__":
    # 打印数学原理
    print("=" * 70)
    print("残差神经网络(ResNet) - PyTorch实现")
    print("=" * 70)
    mathematical_explanation()
    
    # 比较不同架构
    compare_architectures()
    
    # 训练示例（在CIFAR-10上）
    print("\n是否在CIFAR-10上训练ResNet? (需要下载数据集)")
    response = input("输入 'y' 开始训练，或按Enter跳过: ")
    
    if response.lower() == 'y':
        model, history = example_cifar10()
    else:
        print("\n跳过训练。您可以直接运行此脚本并输入'y'来训练模型。")
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)
