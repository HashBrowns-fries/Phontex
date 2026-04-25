"""训练脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ipa_ocr.train import TrainConfig, train_with_config
from ipa_ocr.dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="训练IPA OCR模型")

    # 数据参数
    parser.add_argument(
        "--train-dir", type=str, default="data/train", help="训练数据目录"
    )
    parser.add_argument("--val-dir", type=str, default="data/val", help="验证数据目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")

    # 模型参数
    parser.add_argument(
        "--backbone", type=str, default="mobilenetv3_small_100", help="CNN backbone"
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM隐藏维度")
    parser.add_argument("--num-lstm-layers", type=int, default=3, help="LSTM层数")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout率")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--image-height", type=int, default=64, help="图像高度")
    parser.add_argument("--image-width", type=int, default=256, help="图像宽度")

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 创建配置
    config = TrainConfig(
        train_data_dir=args.train_dir,
        val_data_dir=args.val_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
        use_amp=True,
        patience=20,
    )

    print("=" * 50)
    print("IPA OCR 训练配置")
    print("=" * 50)
    print(f"训练数据: {config.train_data_dir}")
    print(f"验证数据: {config.val_data_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"Backbone: {config.backbone}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"LSTM Layers: {config.num_lstm_layers}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print("=" * 50)

    # 创建数据加载器
    print("加载数据...")
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=(config.image_height, config.image_width),
    )

    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")

    # 开始训练
    print("\n开始训练...\n")
    train_with_config(config, train_loader, val_loader)


if __name__ == "__main__":
    main()
