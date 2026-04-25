"""IPA OCR Engine - 核心识别引擎"""

import torch
from PIL import Image
from typing import Optional, Union, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPAOCREngine:
    """IPA OCR识别引擎"""

    def __init__(self, model: str = "pix2tex"):
        """
        初始化OCR引擎

        Args:
            model: 使用的模型 ("pix2tex" 或 "easyocr")
        """
        self.model_name = model
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self._device}")

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载模型"""
        if self.model_name == "pix2tex":
            self._load_pix2tex()
        elif self.model_name == "easyocr":
            self._load_easyocr()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_pix2tex(self):
        """加载LaTeX-OCR模型"""
        try:
            from pix2tex.cli import LatexOCR

            logger.info("Loading LaTeX-OCR model...")
            self._model = LatexOCR()
            logger.info("LaTeX-OCR model loaded successfully")
        except ImportError:
            logger.error("pix2tex not installed. Run: uv add pix2tex")
            raise

    def _load_easyocr(self):
        """加载EasyOCR模型"""
        try:
            import easyocr

            logger.info("Loading EasyOCR model...")
            self._model = easyocr.Reader(
                ['en'],
                gpu=self._device == "cuda",
                verbose=False
            )
            logger.info("EasyOCR model loaded successfully")
        except ImportError:
            logger.error("easyocr not installed. Run: uv add easyocr")
            raise

    def recognize(
        self,
        image: Union[str, Image.Image],
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        识别图像中的IPA符号

        Args:
            image: 图像路径或PIL Image对象
            return_confidence: 是否返回置信度

        Returns:
            识别结果文本，或(文本, 置信度)元组
        """
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image

        if self.model_name == "pix2tex":
            result = self._recognize_pix2tex(img)
        elif self.model_name == "easyocr":
            result = self._recognize_easyocr(img, return_confidence)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return result

    def _recognize_pix2tex(self, img: Image.Image) -> str:
        """使用LaTeX-OCR识别"""
        try:
            result = self.model(img)
            # 清理LaTeX格式，提取纯文本
            result = self._cleanup_latex(result)
            return result
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return ""

    def _recognize_easyocr(
        self,
        img: Image.Image,
        return_confidence: bool
    ) -> Union[str, Tuple[str, float]]:
        """使用EasyOCR识别"""
        try:
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # EasyOCR需要numpy数组
            import numpy as np
            img_array = np.array(img)

            results = self.model.readtext(img_array)

            if not results:
                return ("", 0.0) if return_confidence else ""

            # 合并所有识别结果
            texts = []
            confidences = []

            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)

            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            if return_confidence:
                return full_text, avg_confidence
            return full_text

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return ("", 0.0) if return_confidence else ""

    def _cleanup_latex(self, latex: str) -> str:
        """清理LaTeX格式，提取相关符号"""
        # 移除常见的LaTeX包装
        import re

        # 保留基本数学符号
        cleaned = latex.strip()

        # 如果是数学环境，提取内容
        if cleaned.startswith('$') and cleaned.endswith('$'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith('\\(') and cleaned.endswith('\\)'):
            cleaned = cleaned[2:-2]
        elif cleaned.startswith('\\[') and cleaned.endswith('\\]'):
            cleaned = cleaned[2:-2]

        return cleaned


def recognize_ipa(
    image_path: str,
    model: str = "pix2tex",
    return_confidence: bool = False
) -> Union[str, Tuple[str, float]]:
    """
    便捷函数：识别IPA符号

    Args:
        image_path: 图像文件路径
        model: 使用的模型
        return_confidence: 是否返回置信度

    Returns:
        识别结果
    """
    engine = IPAOCREngine(model=model)
    return engine.recognize(image_path, return_confidence=return_confidence)
