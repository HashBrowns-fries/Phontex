"""推理模块"""

import torch
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
from PIL import Image

from ipa_ocr.model import create_model
from ipa_ocr.dataset.augment import get_val_transforms
from ipa_ocr.train.config import TrainConfig
from ipa_ocr.utils.characters import idx_to_char


class IPAPredictor:
    """IPA OCR预测器"""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            config: 训练配置（如果为None则从checkpoint加载）
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载配置
        if config is None:
            config_dict = checkpoint.get('config', {})
            config = TrainConfig(**config_dict)

        self.config = config
        self.decode_method = config.decode_method

        # 创建模型
        self.model = create_model(
            backbone=config.backbone,
            hidden_dim=config.hidden_dim,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout,
            use_v2=config.use_v2,
            pretrained=False
        ).to(self.device)

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 图像转换
        self.transform = get_val_transforms(
            (config.image_height, config.image_width)
        )

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_probs: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        预测单个图像

        Args:
            image: 图像路径、PIL Image或numpy数组
            return_probs: 是否返回概率

        Returns:
            预测文本，或(文本, 概率)元组
        """
        # 预处理
        input_tensor = self._preprocess(image)

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            log_probs = self.model(input_tensor)

        # 解码
        predictions = self.model.decode(log_probs, method=self.decode_method)
        text = predictions[0] if predictions else ''

        if return_probs:
            # 计算置信度
            probs = torch.exp(log_probs)
            max_probs = probs.max(dim=-1)[0]
            confidence = max_probs.mean().item()
            return text, confidence

        return text

    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]]
    ) -> List[str]:
        """批量预测"""
        results = []

        for image in images:
            text = self.predict(image)
            results.append(text)

        return results

    def _preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """预处理图像"""
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 转换
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 应用变换
        transformed = self.transform(image=image)

        # 转换为张量
        tensor = torch.from_numpy(transformed['image']).float()

        # 添加batch维度
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor


class IPATrainer:
    """IPA训练器 - 简化版训练入口"""

    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.trainer = None

    def prepare_data(self):
        """准备数据"""
        from ipa_ocr.dataset import create_dataloaders

        train_loader, val_loader = create_dataloaders(
            train_dir=self.config.train_data_dir,
            val_dir=self.config.val_data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            image_size=(self.config.image_height, self.config.image_width)
        )

        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        """开始训练"""
        from ipa_ocr.train import Trainer

        self.trainer = Trainer(self.config)
        self.trainer.train(train_loader, val_loader)

        return self.trainer


def load_predictor(
    checkpoint_path: str,
    device: Optional[str] = None
) -> IPAPredictor:
    """加载预测器"""
    return IPAPredictor(checkpoint_path, device=device)
