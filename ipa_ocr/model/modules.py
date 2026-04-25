"""模型模块定义 - 改进版"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CNNBackbone(nn.Module):
    """CNN特征提取器 - MobileNetV3"""

    def __init__(
        self, backbone_name: str = "mobilenetv3_small_100", pretrained: bool = True
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        original_conv = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv_stem.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        if backbone_name == "mobilenetv3_small_100":
            self.feature_dim = 576
        elif backbone_name == "mobilenetv3_small_050":
            self.feature_dim = 256
        elif backbone_name == "mobilenetv3_large_100":
            self.feature_dim = 1280
        else:
            self.feature_dim = 576

    def forward(self, x):
        features = self.backbone(x)
        return features


class ResNetBackbone(nn.Module):
    """ResNet特征提取器 - 支持灰度图"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(pretrained=pretrained)

        # 修改第一层卷积以接受灰度图
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        return features


class BidirectionalLSTM(nn.Module):
    """双向LSTM序列建模"""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        output, _ = self.lstm(x)
        return output


class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(-1).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


class SelfAttention(nn.Module):
    """自注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器：保留高度信息 → (B, 512, H', W')

    输入: (B, 1, 64, 256)
    输出: (B, 512, 4, 64)  高度方向池化4倍，宽度方向池化4倍
    LSTM输入: (B, 64, 512)  每列对应原始图像 ~4像素宽
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        def conv_block(in_ch, out_ch, pool_h=True, pool_w=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool_h and pool_w:
                layers.append(nn.MaxPool2d(2, 2))
            elif pool_h:
                layers.append(nn.MaxPool2d((2, 1), (2, 1)))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64, pool_h=True, pool_w=True)
        self.conv2 = conv_block(64, 128, pool_h=True, pool_w=True)
        self.conv3 = conv_block(128, 256, pool_h=True, pool_w=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.conv2(x)  # (B, 128, H/4, W/4)
        x = self.conv3(x)  # (B, 256, H/8, W/4)
        x = self.conv4(x)  # (B, 512, H/16, W/4)
        return x


class CRNNModel(nn.Module):
    """CRNN模型: CNN + LSTM + CTC (V1 — MobileNetV3 backbone, T=1，不适合CTC)"""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenetv3_small_100",
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = CNNBackbone(backbone, pretrained=pretrained)
        print(f"CNN feature_dim: {self.cnn.feature_dim}")

        self.projection = nn.Sequential(
            nn.Linear(self.cnn.feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.lstm = BidirectionalLSTM(hidden_dim, hidden_dim, num_lstm_layers, dropout)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        seq_features = self.projection(cnn_features)
        seq_features = seq_features.unsqueeze(1)
        lstm_out = self.lstm(seq_features)
        logits = self.classifier(lstm_out)
        logits = logits.squeeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class CRNNModelV2(nn.Module):
    """CRNN模型V2 - 支持变长输入"""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenetv3_small_100",
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.cnn = self._build_cnn(backbone, pretrained)

        self.lstm = BidirectionalLSTM(256, hidden_dim, num_lstm_layers, dropout)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def _build_cnn(self, backbone_name: str, pretrained: bool):
        layers = [
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out = self.lstm(cnn_out)
        logits = self.classifier(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class CRNNModelV3(nn.Module):
    """CRNN模型V3 — V4的早期版本，输出单个向量"""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        self.cnn = CNNFeatureExtractor(in_channels=1)

        self.lstm = BidirectionalLSTM(
            self.cnn.out_channels, hidden_dim, num_lstm_layers, dropout
        )

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim * 2)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()

        if h == 1:
            conv_out = conv_out.squeeze(2)
        else:
            conv_out = conv_out[:, :, 0, :]

        conv_out = conv_out.permute(0, 2, 1)

        lstm_out = self.lstm(conv_out)

        if self.use_attention:
            query = lstm_out[:, -1:, :]
            context, _ = self.attention(query, lstm_out)
            context = context.squeeze(1)
            logits = self.classifier(context)
        else:
            logits = self.classifier(lstm_out[:, -1, :])

        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs


class CRNNModelV4(nn.Module):
    """CRNN模型V4 — CNNFeatureExtractor + 双向LSTM，输出完整序列 → CTC

    修复：移除了重复的forward定义，确保正确取第一行高度特征。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.cnn = CNNFeatureExtractor(in_channels=1)

        self.lstm = BidirectionalLSTM(
            self.cnn.out_channels, hidden_dim, num_lstm_layers, dropout
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        # 取第一行高度（conv后仍有H'=4，字符高度覆盖在这几行中）
        conv_out = conv_out[:, :, 0, :]  # (B, 512, W')
        conv_out = conv_out.permute(0, 2, 1)  # (B, W', 512)

        lstm_out = self.lstm(conv_out)  # (B, W', 512)
        logits = self.classifier(lstm_out)  # (B, W', num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs  # (B, T, C)


class CRNNModelV5(nn.Module):
    """CRNN模型V5 — 改进版，专为IPA字符识别设计

    改进点：
    - CNN最后做 1×H 自适应池化，保留宽度方向所有信息
    - 更深的LSTM（3层）
    - BatchNorm在CNN后 + Dropout
    - 分类器用2层MLP
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_lstm_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = CNNFeatureExtractorV5(in_channels=1)

        self.lstm = BidirectionalLSTM(
            512, hidden_dim, num_lstm_layers, dropout
        )

        self.bn_lstm = nn.BatchNorm1d(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        conv_out = self.cnn(x)  # (B, 512, 1, W')
        b, c, h, w = conv_out.size()
        conv_out = conv_out.squeeze(2)  # (B, 512, W')
        conv_out = conv_out.permute(0, 2, 1)  # (B, W', 512)

        lstm_out = self.lstm(conv_out)  # (B, W', 512)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, 512, W')
        lstm_out = self.bn_lstm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, W', 512)

        logits = self.classifier(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs  # (B, T, C)


class CNNFeatureExtractorV5(nn.Module):
    """V5专用CNN：保持宽度信息，高度方向做自适应池化到1"""

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # H→1, 保留W
        )

        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x  # (B, 512, 1, W')
