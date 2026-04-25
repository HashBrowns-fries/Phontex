"""测试模型训练"""

import sys
sys.path.insert(0, '.')

import torch
from ipa_ocr.model import create_model
from ipa_ocr.dataset import create_dataloaders
from ipa_ocr.train import TrainConfig, train_with_config


def test_model():
    """测试模型创建和前向传播"""
    print("=" * 50)
    print("测试模型创建")
    print("=" * 50)

    # 创建模型
    model = create_model(
        backbone='mobilenetv3_small_100',
        hidden_dim=256,
        num_lstm_layers=2,
        use_v2=True,
        pretrained=False
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 1, 64, 256)
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过!")

    return model


def test_dataloader():
    """测试数据加载"""
    print("\n" + "=" * 50)
    print("测试数据加载")
    print("=" * 50)

    train_loader, val_loader = create_dataloaders(
        train_dir='data/train',
        val_dir='data/val',
        batch_size=4,
        num_workers=0,
        image_size=(64, 256)
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 获取一个批次
    batch = next(iter(train_loader))
    images, targets, target_lengths = batch

    print(f"图像批次: {images.shape}")
    print(f"目标批次: {targets.shape}")
    print(f"目标长度: {target_lengths}")

    print("数据加载测试通过!")

    return train_loader, val_loader


def test_training():
    """测试训练流程"""
    print("\n" + "=" * 50)
    print("测试训练流程")
    print("=" * 50)

    # 创建配置 - 使用v2模型（自定义CNN）
    config = TrainConfig(
        train_data_dir='data/train',
        val_data_dir='data/val',
        output_dir='outputs',
        backbone='mobilenetv3_small_100',
        hidden_dim=128,
        num_lstm_layers=1,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,
        device='cpu',
        num_workers=0,
        log_interval=5,
        eval_interval=1,
        use_v2=True,  # 使用自定义CNN模型
    )

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=(config.image_height, config.image_width)
    )

    # 训练
    train_with_config(config, train_loader, val_loader)

    print("训练测试完成!")


if __name__ == "__main__":
    test_model()
    test_dataloader()
    test_training()
