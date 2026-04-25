"""IPA数据集定义"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ipa_ocr.utils.characters import char_to_idx, idx_to_char
from ipa_ocr.dataset.augment import get_train_transforms, get_val_transforms


class IPADataset(Dataset):
    """IPA OCR数据集"""

    def __init__(
        self,
        data_dir: str,
        char_to_idx: Dict[str, int] = char_to_idx,
        transform=None,
        max_length: int = 50,
        is_train: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录，包含images和labels.json
            char_to_idx: 字符到索引的映射
            transform: 数据增强
            max_length: 最大文本长度
            is_train: 是否训练集
        """
        self.data_dir = Path(data_dir)
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_length = max_length
        self.is_train = is_train

        # 加载标签
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str]]:
        """加载样本"""
        samples = []

        # 查找labels文件
        label_paths = [
            self.data_dir / "labels.json",
            self.data_dir / "labels.txt",
            self.data_dir / "annotations.json",
        ]

        label_path = None
        for p in label_paths:
            if p.exists():
                label_path = p
                break

        if label_path and label_path.suffix == ".json":
            with open(label_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
                for img_name, text in labels.items():
                    img_path = self.data_dir / "images" / img_name
                    if img_path.exists():
                        samples.append((str(img_path), text))
        elif label_path and label_path.suffix == ".txt":
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_name, text = parts[0], parts[1]
                        img_path = self.data_dir / "images" / img_name
                        if img_path.exists():
                            samples.append((str(img_path), text))

        # 如果没有标签文件，从目录结构加载
        if not samples:
            img_dir = self.data_dir / "images"
            if img_dir.exists():
                for img_path in sorted(img_dir.glob("*.png")):
                    # 使用文件名作为标签（临时）
                    text = img_path.stem
                    samples.append((str(img_path), text))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """获取样本"""
        img_path, text = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert("L")  # 灰度图
        image = np.array(image)

        # 裁切掉四周空白，保留文字内容（解决合成图大量留白问题）
        from ipa_ocr.dataset.augment import crop_content_bbox
        bbox = crop_content_bbox(image, threshold=245, pad=4)
        if bbox is not None:
            y1, y2, x1, x2 = bbox
            image = image[y1:y2, x1:x2]

        # 应用增强（resize + 归一化）
        if self.transform:
            image = self.transform(image=image)["image"]
            # 转换为张量
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
                elif image.dim() == 3:
                    image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        else:
            # 默认转换
            image = torch.from_numpy(image).float() / 255.0
            if image.dim() == 2:
                image = image.unsqueeze(0)

        # 编码文本
        encoded = self._encode_text(text)

        return image, encoded, len(text)

    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本为索引 - CTC不需要填充"""
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                indices.append(0)  # 未知字符暂时用blank
        return torch.tensor(indices, dtype=torch.long)


class IPAGeneratorDataset(Dataset):
    """从IPA文本生成图像的数据集"""

    def __init__(
        self,
        ipa_texts: List[str],
        font_paths: List[str],
        char_to_idx: Dict[str, int] = char_to_idx,
        transform=None,
        max_length: int = 50,
        image_size: Tuple[int, int] = (64, 256),
    ):
        """
        Args:
            ipa_texts: IPA文本列表
            font_paths: 字体路径列表
            char_to_idx: 字符映射
            transform: 数据增强
            max_length: 最大长度
            image_size: 图像尺寸 (H, W)
        """
        self.ipa_texts = ipa_texts
        self.font_paths = font_paths
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ipa_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """生成样本"""
        text = self.ipa_texts[idx]
        font_path = self.font_paths[idx % len(self.font_paths)]

        # 生成图像
        image = self._render_text(text, font_path)

        # 应用增强
        if self.transform:
            image = self.transform(image=image)["image"]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
                if image.dim() == 2:
                    image = image.unsqueeze(0)
        else:
            image = torch.from_numpy(image).float() / 255.0
            if image.dim() == 2:
                image = image.unsqueeze(0)

        # 编码
        encoded = self._encode_text(text)

        return image, encoded, len(text)

    def _render_text(self, text: str, font_path: str) -> np.ndarray:
        """渲染文本为图像"""
        from PIL import Image, ImageDraw, ImageFont

        width, height = self.image_size
        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(font_path, 48)
        except:
            font = ImageFont.load_default()

        # 测量文本
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 居中
        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), text, fill=0, font=font)

        return np.array(image)

    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        indices = [self.char_to_idx.get(c, 0) for c in text]
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (64, 256),
):
    """创建数据加载器"""
    from torch.utils.data import DataLoader

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = IPADataset(train_dir, transform=train_transform, is_train=True)

    val_dataset = IPADataset(val_dir, transform=val_transform, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def collate_fn(batch):
    """批处理整理函数"""
    images, targets, target_lengths = zip(*batch)

    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths
