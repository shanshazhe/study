"""
通用自定义 Dataset 模块

支持多种数据加载方式:
1. ImageFolderDataset - 从 class/images 目录结构加载
2. ImageListDataset - 从图片路径列表加载
3. CSVDataset - 从 CSV 文件加载 (path, label)
4. MemoryDataset - 从内存中的 numpy/tensor 加载

用法示例见文件底部
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import csv


class ImageListDataset(Dataset):
    """
    从图片路径列表创建数据集
    
    Args:
        image_paths: 图片路径列表
        labels: 标签列表 (可选，无监督任务可不传)
        transform: 图像变换
    """
    
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        if labels is not None and len(image_paths) != len(labels):
            raise ValueError("image_paths 和 labels 长度必须一致")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image, -1  # 无标签时返回 -1


class ImageFolderDataset(Dataset):
    """
    从目录结构加载数据集
    
    支持两种结构:
    1. class 子目录结构: root/class_a/*.jpg, root/class_b/*.jpg
    2. 扁平结构: root/*.jpg (所有图片标签为0)
    
    Args:
        root: 根目录路径
        transform: 图像变换
        extensions: 支持的图片扩展名
    """
    
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    def __init__(self, root, transform=None, extensions=None):
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.EXTENSIONS
        
        self.samples = []  # [(path, label), ...]
        self.classes = []
        self.class_to_idx = {}
        
        self._load_samples()
    
    def _load_samples(self):
        # 检查是否有子目录
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]
        
        if subdirs:
            # class 子目录结构
            self.classes = sorted([d.name for d in subdirs])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            
            for class_name in self.classes:
                class_dir = self.root / class_name
                label = self.class_to_idx[class_name]
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in self.extensions:
                        self.samples.append((img_path, label))
        else:
            # 扁平结构
            self.classes = ['default']
            self.class_to_idx = {'default': 0}
            for img_path in self.root.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, 0))
        
        if not self.samples:
            raise ValueError(f"未找到图片: {self.root}")
        
        print(f"加载 {len(self.samples)} 张图片, {len(self.classes)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CSVDataset(Dataset):
    """
    从 CSV 文件加载数据集
    
    CSV 格式: image_path,label 或 image_path,label,... (多列)
    
    Args:
        csv_path: CSV 文件路径
        root: 图片根目录 (如果 CSV 中是相对路径)
        transform: 图像变换
        path_col: 图片路径列索引或列名 (默认 0)
        label_col: 标签列索引或列名 (默认 1)
        has_header: CSV 是否有表头
    """
    
    def __init__(self, csv_path, root=None, transform=None, 
                 path_col=0, label_col=1, has_header=True):
        self.root = Path(root) if root else None
        self.transform = transform
        
        self.samples = []
        self._load_csv(csv_path, path_col, label_col, has_header)
    
    def _load_csv(self, csv_path, path_col, label_col, has_header):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            if has_header:
                header = next(reader)
                # 支持列名
                if isinstance(path_col, str):
                    path_col = header.index(path_col)
                if isinstance(label_col, str):
                    label_col = header.index(label_col)
            
            for row in reader:
                img_path = row[path_col]
                label = row[label_col]
                
                # 尝试转换标签为整数
                try:
                    label = int(label)
                except ValueError:
                    pass  # 保持字符串标签
                
                if self.root:
                    img_path = self.root / img_path
                
                self.samples.append((img_path, label))
        
        # 如果标签是字符串，转换为整数索引
        labels = [s[1] for s in self.samples]
        if labels and isinstance(labels[0], str):
            unique_labels = sorted(set(labels))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            self.samples = [(p, self.label_to_idx[l]) for p, l in self.samples]
            print(f"标签映射: {self.label_to_idx}")
        
        print(f"从 CSV 加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MemoryDataset(Dataset):
    """
    从内存数据创建数据集
    
    Args:
        images: numpy array 或 torch tensor, shape (N, H, W, C) 或 (N, C, H, W)
        labels: numpy array 或 list
        transform: 图像变换 (可选，通常数据已预处理)
        channel_first: images 是否为 channel first 格式
    """
    
    def __init__(self, images, labels=None, transform=None, channel_first=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.channel_first = channel_first
        
        # 转为 tensor
        if not isinstance(self.images, torch.Tensor):
            self.images = torch.from_numpy(self.images).float()
        
        # 归一化到 [0, 1]
        if self.images.max() > 1.0:
            self.images = self.images / 255.0
        
        # 调整维度顺序
        if not channel_first and self.images.dim() == 4:
            self.images = self.images.permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            # 转为 PIL 再变换
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            return image, label
        
        return image, -1


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
    """创建 DataLoader 的便捷函数"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


def get_default_transform(image_size=224, is_train=True):
    """获取默认的图像变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


# ============ 使用示例 ============
if __name__ == "__main__":
    print("=" * 50)
    print("自定义 Dataset 使用示例")
    print("=" * 50)
    
    # 示例 1: 从路径列表创建
    print("\n[1] ImageListDataset 示例:")
    print("""
    paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    labels = [0, 1, 0]
    transform = get_default_transform(224, is_train=True)
    
    dataset = ImageListDataset(paths, labels, transform)
    loader = create_dataloader(dataset, batch_size=2)
    """)
    
    # 示例 2: 从文件夹创建
    print("\n[2] ImageFolderDataset 示例:")
    print("""
    # 目录结构:
    # data/train/
    #   ├── cat/
    #   │   ├── cat1.jpg
    #   │   └── cat2.jpg
    #   └── dog/
    #       ├── dog1.jpg
    #       └── dog2.jpg
    
    dataset = ImageFolderDataset('data/train', transform=get_default_transform(224))
    print(dataset.classes)  # ['cat', 'dog']
    print(dataset.class_to_idx)  # {'cat': 0, 'dog': 1}
    """)
    
    # 示例 3: 从 CSV 创建
    print("\n[3] CSVDataset 示例:")
    print("""
    # labels.csv:
    # image_path,label
    # train/001.jpg,cat
    # train/002.jpg,dog
    
    dataset = CSVDataset(
        csv_path='labels.csv',
        root='data/',
        transform=get_default_transform(224)
    )
    """)
    
    # 示例 4: 从内存数据创建
    print("\n[4] MemoryDataset 示例:")
    print("""
    import numpy as np
    
    images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, 100)
    
    dataset = MemoryDataset(images, labels)
    loader = create_dataloader(dataset, batch_size=16)
    """)
    
    # 完整训练示例
    print("\n[完整训练流程示例]:")
    print("""
    from custom_dataset import ImageFolderDataset, create_dataloader, get_default_transform
    
    # 1. 创建数据集
    train_dataset = ImageFolderDataset(
        'data/train',
        transform=get_default_transform(224, is_train=True)
    )
    test_dataset = ImageFolderDataset(
        'data/test', 
        transform=get_default_transform(224, is_train=False)
    )
    
    # 2. 创建 DataLoader
    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. 训练循环
    for epoch in range(10):
        for images, labels in train_loader:
            # 前向传播、反向传播...
            pass
    """)
