"""生成IPA测试图像"""

from PIL import Image, ImageDraw, ImageFont
import os

# IPA测试字符
IPA_TEST_STRINGS = [
    "ə æ ʃ θ ŋ iː ʊ",
    "p b t d k g f v",
    "s z ʒ h m n l r",
    "j w ɒ ɑː ɔː ɜː",
    "i ɪ u ʊ e ʌ ɔ",
]


def create_ipa_image(text: str, output_path: str, font_size: int = 48):
    """创建IPA测试图像"""
    # 创建白色背景图像
    width, height = 800, 200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # 尝试加载系统字体
    try:
        # Windows上的字体路径
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/seguiemj.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 绘制文本
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill='black', font=font)

    # 保存图像
    image.save(output_path)
    print(f"Created: {output_path}")


def main():
    """生成所有测试图像"""
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)

    for i, ipa_text in enumerate(IPA_TEST_STRINGS):
        output_path = os.path.join(output_dir, f"ipa_test_{i+1}.png")
        create_ipa_image(ipa_text, output_path)

    print(f"\n生成了 {len(IPA_TEST_STRINGS)} 个测试图像")


if __name__ == "__main__":
    main()
