"""
通用数据集均值和标准差计算工具

用于计算图像数据集各通道的 mean 和 std，供 transforms.Normalize 使用

用法:
    # 命令行 - 从文件夹计算
    python calculate_dataset_stats.py /path/to/dataset
    python calculate_dataset_stats.py /path/to/dataset --size 224
    
    # 代码中使用
    from calculate_dataset_stats import calculate_mean_std, calculate_from_folder
    mean, std = calculate_from_folder('/path/to/dataset')
"""

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image


class SimpleImageDataset(Dataset):
    """简单图片数据集，支持单层文件夹（无子目录分类）"""
    
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.image_paths = [
            p for p in self.folder_path.rglob('*') 
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.image_paths:
            raise ValueError(f"未找到图片文件: {folder_path}")
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # 返回 (image, dummy_label)


def calculate_mean_std(dataset, batch_size=256, num_workers=0):
    """
    计算数据集的均值和标准差
    
    Args:
        dataset: PyTorch Dataset 对象，需返回 (image_tensor, label)
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        mean: 各通道均值 (C,)
        std: 各通道标准差 (C,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=False, num_workers=num_workers)
    
    # 获取通道数
    sample = dataset[0][0]
    num_channels = sample.shape[0]
    
    # 累计统计量
    pixel_count = 0
    channel_sum = torch.zeros(num_channels)
    channel_sum_sq = torch.zeros(num_channels)
    
    print("计算中...")
    for images, _ in loader:
        # images shape: (B, C, H, W)
        batch_size_actual = images.shape[0]
        pixels_per_image = images.shape[2] * images.shape[3]
        pixel_count += batch_size_actual * pixels_per_image
        
        # 对每个通道求和
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    
    # 计算均值和标准差
    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_sum_sq / pixel_count - mean ** 2)
    
    return mean, std


def calculate_from_folder(folder_path, image_size=None, use_imagefolder=True):
    """
    从图片文件夹计算均值和标准差
    
    Args:
        folder_path: 图片文件夹路径
        image_size: 可选，调整图片大小 (int 或 tuple)
        use_imagefolder: True 使用 ImageFolder (需 class/images 结构)
                        False 使用 SimpleImageDataset (支持任意结构)
    
    Returns:
        mean, std
    """
    transform_list = []
    if image_size:
        size = (image_size, image_size) if isinstance(image_size, int) else image_size
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    if use_imagefolder:
        try:
            dataset = ImageFolder(folder_path, transform=transform)
            print(f"使用 ImageFolder 加载，找到 {len(dataset)} 张图片")
        except Exception:
            print("ImageFolder 加载失败，尝试 SimpleImageDataset...")
            dataset = SimpleImageDataset(folder_path, transform=transform)
    else:
        dataset = SimpleImageDataset(folder_path, transform=transform)
    
    return calculate_mean_std(dataset)


def format_output(mean, std):
    """格式化输出结果"""
    num_channels = len(mean)
    
    if num_channels == 1:
        mean_str = f"({mean[0]:.4f},)"
        std_str = f"({std[0]:.4f},)"
    else:
        mean_str = "(" + ", ".join(f"{v:.4f}" for v in mean.tolist()) + ")"
        std_str = "(" + ", ".join(f"{v:.4f}" for v in std.tolist()) + ")"
    
    print(f"\nMean: {mean_str}")
    print(f"Std:  {std_str}")
    print(f"\n用法: transforms.Normalize({mean_str}, {std_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算图像数据集的均值和标准差")
    parser.add_argument("folder", help="图片文件夹路径")
    parser.add_argument("--size", type=int, default=None, 
                        help="统一调整图片大小 (如 224)")
    parser.add_argument("--simple", action="store_true",
                        help="使用简单模式 (无需 class/images 目录结构)")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("数据集均值和标准差计算工具")
    print("=" * 50)
    print(f"文件夹: {args.folder}")
    if args.size:
        print(f"图片大小: {args.size}x{args.size}")
    
    mean, std = calculate_from_folder(
        args.folder, 
        image_size=args.size,
        use_imagefolder=not args.simple
    )
    format_output(mean, std)
