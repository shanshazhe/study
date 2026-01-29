import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def ResNet18(num_classes=10):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx+1}, Loss: {running_loss/100:.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total


def validate(model, test_loader, criterion, device):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.3f}, Test Acc: {accuracy:.2f}%')
    
    return accuracy


def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.1
    num_classes = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
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
    
    # Load CIFAR-10 dataset
    print('Loading CIFAR-10 dataset...')
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    # Create model
    print('Building ResNet-18 model...')
    model = ResNet18(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print('Starting training...')
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('saved_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'saved_models/resnet18_best.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')


def load_model(model_path='saved_models/resnet18_best.pth', num_classes=10, device=None):
    """加载已保存的模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ResNet18(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功: {model_path}")
    print(f"训练时最佳准确率: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    
    return model


def predict(model, images, device=None):
    """
    对图像进行预测
    
    Args:
        model: 加载的模型
        images: 图像张量 (B, C, H, W) 或 (C, H, W)
        device: 设备
    
    Returns:
        predictions: 预测类别
        probabilities: 预测概率
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 确保是 batch 格式
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions, probabilities


def predict_and_show(model_path='saved_models/resnet18_best.pth', num_samples=10):
    """加载模型并在测试集上预测，打印结果"""
    
    # CIFAR-10 类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    model = load_model(model_path, num_classes=10, device=device)
    
    # 加载测试数据
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    # 预测前 num_samples 个样本
    print(f"\n{'='*60}")
    print(f"预测结果 (前 {num_samples} 个样本)")
    print(f"{'='*60}")
    print(f"{'序号':<6}{'真实标签':<15}{'预测标签':<15}{'置信度':<10}{'结果'}")
    print('-' * 60)
    
    correct = 0
    for i in range(num_samples):
        image, label = test_dataset[i]
        pred, prob = predict(model, image, device)
        
        pred_label = pred.item()
        confidence = prob[0, pred_label].item() * 100
        is_correct = pred_label == label
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{i:<6}{classes[label]:<15}{classes[pred_label]:<15}{confidence:>6.2f}%   {status}")
    
    print('-' * 60)
    print(f"准确率: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # 预测模式: python resnet_example_old.py predict
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        predict_and_show(num_samples=num_samples)
    else:
        # 训练模式: python resnet_example_old.py
        main()
