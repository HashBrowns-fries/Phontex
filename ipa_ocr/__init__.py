"""IPA OCR识别系统"""

__version__ = "0.1.0"

from ipa_ocr.model import create_model, IPAOCRModel
from ipa_ocr.dataset import IPADataset, IPAGeneratorDataset
from ipa_ocr.train import TrainConfig, Trainer, train_with_config
from ipa_ocr.infer import IPAPredictor, load_predictor
from ipa_ocr.engine import IPAOCREngine, recognize_ipa

__all__ = [
    '__version__',
    'create_model',
    'IPAOCRModel',
    'IPADataset',
    'IPAGeneratorDataset',
    'TrainConfig',
    'Trainer',
    'train_with_config',
    'IPAPredictor',
    'load_predictor',
    'IPAOCREngine',
    'recognize_ipa',
]
