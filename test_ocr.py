"""测试IPA OCR识别"""

from ipa_ocr.engine import IPAOCREngine


def test_easyocr():
    """测试EasyOCR模型"""
    print("=" * 50)
    print("Testing EasyOCR Model")
    print("=" * 50)

    engine = IPAOCREngine(model="easyocr")

    import os
    test_dir = "test_images"

    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(test_dir, filename)
            print(f"\nTesting: {filename}")

            result = engine.recognize(filepath, return_confidence=True)

            if isinstance(result, tuple):
                text, confidence = result
                print(f"  Result: {text}")
                print(f"  Confidence: {confidence:.2%}")
            else:
                print(f"  Result: {result}")


def test_pix2tex():
    """测试LaTeX-OCR模型"""
    print("\n" + "=" * 50)
    print("Testing LaTeX-OCR (pix2tex) Model")
    print("=" * 50)

    try:
        engine = IPAOCREngine(model="pix2tex")

        import os
        test_dir = "test_images"

        for filename in sorted(os.listdir(test_dir)):
            if filename.endswith('.png'):
                filepath = os.path.join(test_dir, filename)
                print(f"\nTesting: {filename}")

                result = engine.recognize(filepath)
                print(f"  Result: {result}")

    except Exception as e:
        print(f"pix2tex test error: {e}")


if __name__ == "__main__":
    test_easyocr()
    test_pix2tex()
