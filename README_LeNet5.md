# LeNet-5 PyTorch Implementation

## Overview
LeNet-5 is a classic convolutional neural network (CNN) architecture designed by Yann LeCun et al. in 1998. It was originally designed for handwritten digit recognition and achieves excellent results on the MNIST dataset.

## Architecture

```
INPUT (32×32) 
    ↓
CONV1 (6 filters, 5×5) → ReLU → AVG_POOL (2×2)
    ↓
CONV2 (16 filters, 5×5) → ReLU → AVG_POOL (2×2)
    ↓
FLATTEN
    ↓
FC1 (120 neurons) → ReLU
    ↓
FC2 (84 neurons) → ReLU
    ↓
FC3 (10 neurons) → OUTPUT
```

**Total Parameters**: ~61,706

## Features

- ✅ Classic LeNet-5 architecture implementation
- ✅ MNIST dataset support with automatic download
- ✅ Training and evaluation functions
- ✅ Model checkpointing (saves best model)
- ✅ Support for CUDA, MPS (Apple Silicon), and CPU
- ✅ Progress tracking and metrics display
- ✅ Inference demo function

## Usage

### 1. Train the Model

Simply run the script to start training:

```bash
python leNet5.py
```

This will:
- Download the MNIST dataset (if not already downloaded)
- Train for 10 epochs
- Evaluate on test set after each epoch
- Save the best model as `lenet5_best.pth`

### 2. Customize Hyperparameters

Edit these variables in the `main()` function:

```python
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 10            # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
```

### 3. Use the Model in Your Code

```python
from leNet5 import LeNet5
import torch

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5(num_classes=10).to(device)

# Load trained weights
model.load_state_dict(torch.load('lenet5_best.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(your_input_tensor)
    predictions = output.argmax(dim=1)
```

### 4. Run Inference Demo

Uncomment the last lines in `leNet5.py`:

```python
if __name__ == '__main__':
    main()
    
    # Uncomment these lines:
    print("\n" + "="*70)
    print("Running inference demo...")
    print("="*70 + "\n")
    demo_inference()
```

## Expected Performance

On MNIST dataset, LeNet-5 typically achieves:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98-99%
- **Training Time**: ~2-5 minutes on modern hardware

## Requirements

```
torch>=2.9.1
torchvision>=0.24.1
```

## Model Details

### Layer Specifications

1. **Conv1**: 1→6 channels, 5×5 kernel, padding=2
2. **Pool1**: Average pooling, 2×2 kernel
3. **Conv2**: 6→16 channels, 5×5 kernel
4. **Pool2**: Average pooling, 2×2 kernel
5. **FC1**: 400→120 neurons
6. **FC2**: 120→84 neurons
7. **FC3**: 84→10 neurons (output)

### Activation Functions
- **Hidden layers**: ReLU
- **Output layer**: None (logits for CrossEntropyLoss)

## Notes

- The original LeNet-5 used Tanh activation, but this implementation uses ReLU for better performance
- Input images are resized to 32×32 (original LeNet-5 input size)
- MNIST images are normalized with mean=0.1307 and std=0.3081

## References

- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - Original LeNet-5 paper by Yann LeCun et al.

