"""训练配置"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class TrainConfig:
    """训练配置"""

    # 模型参数
    backbone: str = "mobilenetv3_small_100"
    hidden_dim: int = 256
    num_lstm_layers: int = 3
    dropout: float = 0.3
    use_v2: bool = False
    use_v3: bool = False
    use_v4: bool = False
    use_v5: bool = True
    use_attention: bool = True

    # 数据参数
    image_height: int = 64
    image_width: int = 256
    max_length: int = 50

    # 训练参数
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5

    # 数据路径
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    output_dir: str = "outputs"

    # 优化器
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    scheduler_params: dict = field(
        default_factory=lambda: {"T_max": 100, "eta_min": 1e-6}
    )

    # 早停
    patience: int = 15
    min_delta: float = 0.005

    # 设备
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_workers: int = 4

    # 混合精度
    use_amp: bool = True

    # 检查点
    save_best_only: bool = True
    save_interval: int = 5

    # 日志
    log_interval: int = 10
    eval_interval: int = 1

    # 推理
    decode_method: str = "greedy"  # 'greedy' or 'beam_search'

    def __post_init__(self):
        """后处理"""
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 自动检测设备
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model": {
                "backbone": self.backbone,
                "hidden_dim": self.hidden_dim,
                "num_lstm_layers": self.num_lstm_layers,
                "dropout": self.dropout,
                "use_v2": self.use_v2,
                "use_v5": self.use_v5,
            },
            "data": {
                "image_height": self.image_height,
                "image_width": self.image_width,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "train_data_dir": self.train_data_dir,
                "val_data_dir": self.val_data_dir,
            },
            "training": {
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
            },
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "device": self.device,
        }


def get_default_config() -> TrainConfig:
    """获取默认配置"""
    return TrainConfig()


def load_config(config_path: str) -> TrainConfig:
    """从文件加载配置"""
    import json
    from dataclasses import asdict

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return TrainConfig(**config_dict)


def save_config(config: TrainConfig, config_path: str):
    """保存配置"""
    import json
    from dataclasses import asdict

    config_dict = asdict(config)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
