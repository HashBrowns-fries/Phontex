"""CRNN模型封装"""

import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from ipa_ocr.model.modules import CRNNModel, CRNNModelV2, CRNNModelV3, CRNNModelV4, CRNNModelV5
from ipa_ocr.utils.characters import idx_to_char, NUM_CLASSES


class IPAOCRModel(nn.Module):
    """IPA OCR模型封装"""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "mobilenetv3_small_100",
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_v2: bool = False,
        use_v3: bool = False,
        use_v4: bool = False,
        use_v5: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        if use_v5:
            self.model = CRNNModelV5(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
            )
        elif use_v4:
            self.model = CRNNModelV4(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
                use_attention=use_attention,
            )
        elif use_v3:
            self.model = CRNNModelV3(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
                use_attention=use_attention,
            )
        elif use_v2:
            self.model = CRNNModelV2(
                num_classes=num_classes,
                backbone=backbone,
                hidden_dim=hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
                pretrained=pretrained,
            )
        else:
            self.model = CRNNModel(
                num_classes=num_classes,
                backbone=backbone,
                hidden_dim=hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
                pretrained=pretrained,
            )

    def forward(self, x):
        return self.model(x)

    def decode(self, log_probs: torch.Tensor, method: str = "greedy") -> List[str]:
        """
        解码预测结果

        Args:
            log_probs: (T, B, C) or (B, C)
            method: 'greedy' 或 'beam_search'

        Returns:
            解码后的文本列表
        """
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)
            _squeeze_output = True
        else:
            _squeeze_output = False

        log_probs = log_probs.permute(1, 0, 2)

        if method == "greedy":
            predictions = self._greedy_decode(log_probs)
        else:
            predictions = self._beam_search_decode(log_probs)

        results = []
        for pred in predictions:
            text = self._indices_to_text(pred)
            results.append(text)

        return results

    def _greedy_decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """Greedy解码"""
        predictions = log_probs.argmax(dim=-1)

        results = []
        for pred in predictions:
            decoded = []
            prev = 0
            for idx in pred.tolist():
                if idx != prev and idx != 0:
                    decoded.append(idx)
                prev = idx
            results.append(decoded)

        return results

    def _beam_search_decode(
        self, log_probs: torch.Tensor, beam_width: int = 5
    ) -> List[List[int]]:
        """Beam Search解码"""
        results = []

        for lp in log_probs:
            beams = [(0.0, [])]

            for t in range(lp.size(0)):
                all_candidates = []
                for score, indices in beams:
                    probs = lp[t]
                    topk_probs, topk_idx = probs.topk(beam_width)

                    for prob, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
                        new_score = score + prob
                        new_indices = indices + [idx]
                        all_candidates.append((new_score, new_indices))

                beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[
                    :beam_width
                ]

            best = beams[0][1]
            decoded = []
            prev = 0
            for idx in best:
                if idx != prev and idx != 0:
                    decoded.append(idx)
                prev = idx
            results.append(decoded)

        return results

    def _indices_to_text(self, indices: List[int]) -> str:
        """将索引转换为文本"""
        text = ""
        for idx in indices:
            if idx in idx_to_char:
                text += idx_to_char[idx]
        return text


def create_model(
    backbone: str = "mobilenetv3_small_100",
    hidden_dim: int = 256,
    num_lstm_layers: int = 2,
    dropout: float = 0.1,
    pretrained: bool = True,
    use_v2: bool = False,
    use_v3: bool = False,
    use_v4: bool = False,
    use_v5: bool = False,
    use_attention: bool = True,
) -> IPAOCRModel:
    """创建模型"""
    return IPAOCRModel(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        pretrained=pretrained,
        use_v2=use_v2,
        use_v3=use_v3,
        use_v4=use_v4,
        use_v5=use_v5,
        use_attention=use_attention,
    )


class CTCLoss(nn.Module):
    """CTC损失函数封装 — 直接使用 F.ctc_loss 避免设备问题"""

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True
    ):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # log_probs: (B, T, C) → CTC需要 (T, B, C)
        if log_probs.dim() == 3:
            log_probs = log_probs.permute(1, 0, 2)
        elif log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0).permute(1, 0, 2)

        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )
