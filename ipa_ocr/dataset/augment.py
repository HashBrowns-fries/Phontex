"""数据增强"""

import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional
import albumentations as A


def crop_content_bbox(image: np.ndarray, threshold: int = 245, pad: int = 4) -> Optional[Tuple[int, int, int, int]]:
    """检测图像中非空白内容的边界框 (y1, y2, x1, x2)，找不到内容返回None"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image
    mask = gray < threshold
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    # 加padding（像素单位）
    y1 = max(0, y1 - pad)
    y2 = min(gray.shape[0] - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(gray.shape[1] - 1, x2 + pad)
    return (y1, y2 + 1, x1, x2 + 1)  # (y1, y2, x1, x2) in slice notation


class CropContent:
    """裁切掉图像四周的空白区域，保留文字内容"""

    def __init__(self, threshold: int = 245, pad: int = 4):
        self.threshold = threshold
        self.pad = pad

    def __call__(self, image: np.ndarray) -> np.ndarray:
        bbox = crop_content_bbox(image, self.threshold, self.pad)
        if bbox is not None:
            y1, y2, x1, x2 = bbox
            image = image[y1:y2, x1:x2]
        return image


class IPADataAugmentation:
    """IPA图像数据增强"""

    def __init__(
        self,
        rotation: Tuple[float, float] = (-3, 3),
        shear: Tuple[float, float] = (-5, 5),
        scale: Tuple[float, float] = (0.9, 1.1),
        brightness: float = 0.2,
        contrast: float = 0.2,
        blur_prob: float = 0.2,
        noise_prob: float = 0.2,
        motion_blur_prob: float = 0.1
    ):
        self.transforms = A.Compose([
            # 几何变换
            A.Affine(
                rotate=rotation,
                shear=shear,
                scale=scale,
                p=0.5,
                mode=cv2.BORDER_REPLICATE
            ),
            # 模糊
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=blur_prob),
            # 噪声
            A.OneOf([
                A.GaussNoise(var_limit=(5, 20), p=1),
                A.ISONoise(p=1),
            ], p=noise_prob),
            # 亮度对比度
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.3
            ),
            # 运动模糊
            A.MotionBlur(blur_limit=3, p=motion_blur_prob),
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用增强

        Args:
            image: numpy数组 (H, W, C)

        Returns:
            增强后的图像
        """
        return self.transforms(image=image)['image']


class SimpleAugmentation:
    """简单的数据增强（不依赖albumentations）"""

    def __init__(
        self,
        rotation_range: float = 3,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1
    ):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """应用增强"""
        img = img.copy()

        # 随机旋转
        if random.random() < 0.3:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img = img.rotate(angle, fillcolor=255)

        # 随机亮度
        if random.random() < 0.3:
            factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # 随机对比度
        if random.random() < 0.3:
            factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        # 随机模糊
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # 随机噪声（模拟）
        if random.random() < 0.1:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """添加噪声"""
        img_array = np.array(img).astype(np.float32)
        noise = np.random.randn(*img_array.shape) * 10
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


def get_train_transforms(image_size: Tuple[int, int] = (64, 256)):
    """获取训练数据增强：resize + 几何/像素增强 + 归一化"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Affine(
            rotate=(-2, 2),
            shear=(-3, 3),
            scale=(0.95, 1.05),
            p=0.5,
            mode=cv2.BORDER_REPLICATE,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 2), p=1),
        ], p=0.15),
        A.OneOf([
            A.GaussNoise(var_limit=(3.0, 12.0), p=1),
        ], p=0.15),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (64, 256)):
    """获取验证数据增强：resize + 归一化（无随机增强）"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])


def crop_content_np(image: np.ndarray, **kwargs) -> np.ndarray:
    """裁切图像空白边距（适配 albumentations Lambda 接口）"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image
    mask = gray < 245
    if not mask.any():
        return image
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    pad = 4
    y1 = max(0, y1 - pad)
    y2 = min(gray.shape[0], y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(gray.shape[1], x2 + pad)
    return image[y1:y2, x1:x2]
