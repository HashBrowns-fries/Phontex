"""IPA字符集定义"""

# 元音 - 完整IPA元音表（去重）
IPA_VOWELS = list(
    dict.fromkeys(
        [
            # 闭元音
            "i",
            "y",
            "ɨ",
            "ʉ",
            "ɯ",
            "u",
            # 次闭
            "ɪ",
            "ʏ",
            "ʊ",
            # 半闭
            "e",
            "ø",
            "ɘ",
            "ɵ",
            "ɤ",
            "o",
            # 中央
            "ə",
            "e̞",
            "ø̞",
            "ɤ̞",
            "o̞",
            # 半开
            "ɛ",
            "œ",
            "ɜ",
            "ɞ",
            "ʌ",
            "ɔ",
            # 开
            "æ",
            "ɐ",
            "a",
            "ɶ",
            "ä",
            "ɑ",
            "ɒ",
        ]
    )
)

# 辅音 - 完整IPA辅音表（去重）
IPA_CONSONANTS = list(
    dict.fromkeys(
        [
            # 爆破音
            "p",
            "b",
            "p̪",
            "b̪",
            "t",
            "d",
            "t̼",
            "d̼",
            "ʈ",
            "ɖ",
            "c",
            "ɟ",
            "k",
            "g",
            "q",
            "ɢ",
            "ʔ",
            # 鼻音
            "m",
            "m̥",
            "ɱ",
            "ɱ̊",
            "n",
            "n̼",
            "ɳ",
            "ɳ̊",
            "ɲ",
            "ɲ̊",
            "ŋ",
            "ŋ̊",
            "ɴ",
            "ɴ̥",
            # 颤音/闪音
            "ʙ",
            "ʙ̥",
            "r",
            "r̥",
            "ʀ",
            "ʀ̥",
            "ⱱ",
            "ⱱ̟",
            "ɾ",
            "ɾ̥",
            "ɽ",
            "ɽ̊",
            # 擦音
            "ɸ",
            "β",
            "f",
            "v",
            "θ",
            "ð",
            "θ̼",
            "ð̼",
            "s",
            "z",
            "ʃ",
            "ʒ",
            "ʂ",
            "ʐ",
            "ç",
            "ʝ",
            "x",
            "ɣ",
            "χ",
            "ʁ",
            "ħ",
            "ʕ",
            "ʜ",
            "ʢ",
            "h",
            "ɦ",
            # 边擦音
            "ɬ",
            "ɮ",
            "ɭ̊˔",
            "ɭ˔",
            "ʎ̝̊",
            "ʎ̝",
            "ʟ̝̊",
            "ʟ̝",
            # 边近音/边闪音
            "l",
            "l̥",
            "ɭ",
            "ɭ̊",
            "ʎ",
            "ʎ̥",
            "ʟ",
            "ʟ̥",
            "ɺ",
            "ɭ̆",
            "ʎ̆",
            "ʟ̆",
            # 近音
            "ʋ",
            "ʋ̥",
            "ɹ",
            "ɹ̥",
            "ɹ̠̊˔",
            "ɹ̠˔",
            "ɻ",
            "ɻ̊",
            "j",
            "j̊",
            "ɰ",
            "ɰ̊",
            "w",
            "ɥ",
            "ɥ̊",
        ]
    )
)

# 超音段音位
SUPERSEGMENTALS = [
    # 重音
    "ˈ",  # 主重音
    "ˌ",  # 次重音
    # 长短
    "ː",  # 长音
    "ˑ",  # 半长
    # 气流/清浊
    "ʰ",  # 送气
    "ʱ",  # 浊送气
    "ʍ",  # 清唇软腭近音
    # 协同调音
    "ʷ",  # 唇化
    "ʲ",  # 腭化
    "ˠ",  # 软腭化
    "ⁿ",  # 软腭化/鼻化
    # 其他修饰
    "˞",  # 卷舌
    "̃",  # 鼻化
]

# 组合标记（Diacritics）- 完整列表
IPA_DIACRITICS = {
    # 上标标记
    "superscript": [
        "ʷ",
        "ʲ",
        "ʰ",
        "ʱ",
        "˞",
        "̹",
        "̜",
        "̟",
        "̠",
        "̝",
        "̞",
    ],
    # 下标标记
    "subscript": [
        "ⁿ",
        "˞",
        "̚",
        "̩",
        "̤",
        "̰",
        "̽",
        "̯",
        "̻",
        "̺",
        "̼",
    ],
    # 发声态修饰符
    "phonation": [
        "̥",
        "̤",
        "̰",
        "̼",
        "ʰ",
        "ʷ",
        "ʲ",
    ],
    # 舌位修饰符
    "tongue": [
        "̟",
        "̠",
        "̝",
        "̞",
        "̘",
        "̙",
        "̩",
        "̯",
    ],
    # 唇形修饰符
    "labial": [
        "̹",
        "̜",
        "̊",
        "̽",
    ],
    # 重音
    "stress": ["ˈ", "ˌ"],
    # 长短
    "length": ["ː", "ˑ"],
    # 鼻化
    "nasal": ["̃"],
}

# 其他符号（去重）
OTHER_SYMBOLS = list(
    dict.fromkeys(
        [
            ".",  # 音节边界
            " ",  # 空格
            "-",  # 连字符
            "'",  # 重音标记
            "ˌ",  # 次重音
            "ː",  # 长音
        ]
    )
)

# IPA 修饰符（来自训练数据）
IPA_MODIFIERS = [
    "̩",
    "̈",
    "̠",
    "̹",
    "̥",
    "̞",
    "̙",
    "̵",
    "̼",
    "̜",
    "̽",
    "̝",
    "̴",
    "̰",
    "̤",
    "̚",
    "̟",
    "̘",
    "̯",
]

# 完整字符集（去重）
IPA_FULL_SET = list(
    dict.fromkeys(
        IPA_VOWELS + IPA_CONSONANTS + SUPERSEGMENTALS + OTHER_SYMBOLS + IPA_MODIFIERS
    )
)

# 构建映射
BLANK_TOKEN = "<blank>"

char_to_idx = {BLANK_TOKEN: 0}
for i, char in enumerate(IPA_FULL_SET):
    char_to_idx[char] = i + 1

idx_to_char = {v: k for k, v in char_to_idx.items()}

NUM_CLASSES = len(char_to_idx)  # 包括blank


def get_characters():
    """获取字符集信息"""
    return {
        "vowels": IPA_VOWELS,
        "consonants": IPA_CONSONANTS,
        "supersegmentals": SUPERSEGMENTALS,
        "diacritics": IPA_DIACRITICS,
        "other": OTHER_SYMBOLS,
        "full_set": IPA_FULL_SET,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "num_classes": NUM_CLASSES,
    }
