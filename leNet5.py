"""
LeNet-5 Implementation in PyTorch
Classic convolutional neural network architecture designed by Yann LeCun et al. (1998)
Originally designed for handwritten digit recognition (MNIST dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """
    LeNet-5 Architecture:
    INPUT -> CONV1 -> POOL1 -> CONV2 -> POOL2 -> FC1 -> FC2 -> FC3 -> OUTPUT

    Original paper: "Gradient-Based Learning Applied to Document Recognition"
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 16 channels * 6 * 6 = 576
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Layer 1: Convolution + Activation + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch, 6, 16, 16)

        # Layer 2: Convolution + Activation + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch, 16, 6, 6)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Output: (batch, 576)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Output: (batch, 120)
        x = F.relu(self.fc2(x))  # Output: (batch, 84)
        x = self.fc3(x)          # Output: (batch, num_classes)

        return x


def get_data_loaders(batch_size=64, num_workers=2):
    """
    Prepare MNIST dataset with appropriate transformations
    """
    # Transform: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'\nTraining Set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """
    Evaluate the model on test dataset
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test Set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


def main():
    """
    Main training loop
    """
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
    print(f'Using device: {device}\n')

    # Load data
    print('Loading MNIST dataset...')
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}\n')

    # Initialize model
    model = LeNet5(num_classes=10).to(device)
    print('Model architecture:')
    print(model)
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print('Starting training...\n')
    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f'{"="*70}')
        print(f'Epoch {epoch}/{EPOCHS}')
        print(f'{"="*70}')

        # Train
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )

        # Test
        test_loss, test_acc = test(model, device, test_loader, criterion)

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'lenet5_best.pth')
            print(f'✓ Saved new best model with accuracy: {best_accuracy:.2f}%\n')

    print(f'{"="*70}')
    print(f'Training completed!')
    print(f'Best test accuracy: {best_accuracy:.2f}%')
    print(f'{"="*70}')


def demo_inference():
    """
    Demo: Load trained model and make predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')

    # Load model
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load('lenet5_best.pth'))
    model.eval()

    # Load a test sample
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get one sample
    data, label = next(iter(test_loader))
    data = data.to(device)

    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

    print(f'Predicted: {pred.item()}, Actual: {label.item()}')
    print(f'Confidence scores: {F.softmax(output, dim=1)[0]}')


if __name__ == '__main__':
    # Train the model
    main()

    # Uncomment to run inference demo after training
    # print("\n" + "="*70)
    # print("Running inference demo...")
    # print("="*70 + "\n")
    # demo_inference()

