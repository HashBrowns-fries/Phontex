"""IPA OCR CLI - 命令行界面"""

import argparse
import sys
from pathlib import Path

from ipa_ocr.engine import IPAOCREngine


def main():
    parser = argparse.ArgumentParser(
        description="IPA OCR - 识别图像中的国际音标符号"
    )
    parser.add_argument(
        "image",
        help="要识别的图像文件路径"
    )
    parser.add_argument(
        "-m", "--model",
        choices=["pix2tex", "easyocr"],
        default="pix2tex",
        help="使用的OCR模型 (默认: pix2tex)"
    )
    parser.add_argument(
        "-c", "--confidence",
        action="store_true",
        help="显示识别置信度"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 文件不存在: {args.image}", file=sys.stderr)
        sys.exit(1)

    # 创建引擎并识别
    engine = IPAOCREngine(model=args.model)

    print(f"正在使用 {args.model} 模型识别图像...")

    result = engine.recognize(
        image_path,
        return_confidence=args.confidence
    )

    if args.confidence:
        text, confidence = result
        print(f"\n识别结果: {text}")
        print(f"置信度: {confidence:.2%}")
    else:
        print(f"\n识别结果: {result}")


if __name__ == "__main__":
    main()
