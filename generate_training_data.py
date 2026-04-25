"""
Phonologically-correct IPA training data generator.

Syllable theory:
  σ (syllable) = onset(ω) + rime(ρ)
  ρ (rime)     = nucleus(ν) + coda(κ)

Template: C₁(C₂) V₁(V₂) (C₃)(C₄)
  onset  = ∅ | C | C₁C₂ | sC₁C₂     — sonority rising
  nucleus = short V | long V | diphthong | nasalized V
  coda   = ∅ | C | C₁C₂              — sonority falling

Suprasegmental rules:
  ˈ primary   — first heavy syllable / antepenult
  ˌ secondary — penult in 3+ syllable words
  ː length    — only vowels in open (no-coda) syllables
  ̃  nasalization — only vowels, only in open syllables
  ʰ aspiration — only voiceless stops (p t tʃ k), rarely
"""

import os
import json
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHONOLOGICAL INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

# Consonants
STOPS_VL   = ['p',  't',  'tʃ', 'k',  'ʔ']
STOPS_VD   = ['b',  'd',  'dʒ', 'g']
NASALS     = ['m',  'n',  'ŋ']
LIQUIDS_L  = ['l',  'r']
FRICATIVES = ['f',  'v',  'θ',  'ð',  's',  'z',  'ʃ',  'ʒ',
               'x',  'ɣ',  'h',  'ɦ',  'ç',  'ʝ',  'ɬ',  'ɮ']
GLIDES     = ['w',  'j',  'ɥ']

# Sorted by sonority (low → high): stops < nasals < liquids < glides
ALL_CONS = (STOPS_VL + STOPS_VD + FRICATIVES +
            NASALS + LIQUIDS_L + GLIDES)

# Valid 2-consonant onset clusters (sonority rising within cluster)
ONSET_2 = [
    'pl', 'pr', 'bl', 'br',
    'tr', 'dr',
    'kl', 'kr', 'gl', 'gr',
    'tw', 'dw', 'kw',
    'sp', 'st', 'sk',          # s + voiceless stop
    'fl', 'fr',
    'θr', 'sl', 'sn', 'sm', 'sw',
    'ml', 'mr',
]

# Valid 3-consonant onset clusters (s + stop + liquid)
ONSET_3 = ['spr', 'spl', 'str', 'skr', 'skl', 'skw', 'sl']

# Valid 2-consonant coda clusters (sonority falling within cluster)
CODA_2 = [
    # stop + stop
    'pt', 'kt', 'bd', 'ɡd', 'pʔ', 'tʔ', 'kʔ',
    # nasal + stop
    'mp', 'mb', 'nt', 'nd', 'ŋk', 'ɱp',
    # stop + fricative
    'pθ', 'tθ', 'kθ',
    # fricative + stop
    'ft', 'fs', 'sp', 'st', 'sk',
    # liquid + stop
    'lp', 'rp', 'lk', 'rk', 'ld', 'rd', 'lɡ', 'rɡ',
    # nasal + fricative
    'mf', 'ns', 'nz', 'nv', 'ŋx',
    # three-element
    'lts', 'rts', 'kts', 'nts', 'mps', 'ŋks',
]

# Vowels
SHORT_V = ['ɪ', 'ʊ', 'ɛ', 'æ', 'ʌ', 'ɒ', 'ə', 'ɐ', 'e', 'o', 'ɤ', 'ɵ']
LONG_V_MAP = {
    'i': 'iː', 'u': 'uː', 'e': 'eː', 'o': 'oː',
    'ɜ': 'ɜː', 'ɔ': 'ɔː', 'ɑ': 'ɑː', 'æ': 'æː',
    'ɛ': 'ɛː', 'ʌ': 'ʌː',
}
DIPHTHONG = ['aɪ', 'eɪ', 'ɔɪ', 'aʊ', 'oʊ', 'ɪə', 'eə', 'ʊə', 'ɜː']

ALL_SHORT_V = SHORT_V + list(LONG_V_MAP.keys())  # covers base forms


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NUCLEUS / CODA HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _vowel_end(s: str, start: int) -> int:
    """Return index one past the last character of vowel starting at s[start]."""
    if start >= len(s):
        return start
    ch = s[start]
    if ch in LONG_V_MAP:
        return start + 2  # LONG_V_MAP[ch] gives the 2-char form but we consume 2 chars
    if start + 1 < len(s) and s[start:start+2] in DIPHTHONG:
        return start + 2
    return start + 1


def _split_syllable(syl: str):
    """
    Split a bare (no stress) syllable into (onset, nucleus, coda).
    Nucleus is the vowel (or vowel cluster) at the center.
    """
    i = 0
    # onset: consume up to 3 chars
    on_end = i
    if i < len(syl) and syl[i] == 's' and i+1 < len(syl) and syl[i+1] in 'ptkbdɡ':
        on_end = i + 3 if i+2 < len(syl) else i + 2
    elif i < len(syl) and syl[i:i+2] in ONSET_3:
        on_end = i + 3
    elif i+1 < len(syl) and syl[i:i+2] in ONSET_2:
        on_end = i + 2
    elif i < len(syl) and syl[i] in ALL_CONS:
        on_end = i + 1
    onset = syl[i:on_end]

    # nucleus: consume vowel chars
    nucl_start = on_end
    nucl_end = _vowel_end(syl, nucl_start)
    nucleus = syl[nucl_start:nucl_end]

    # coda: rest
    coda = syl[nucl_end:]

    # If nucleus is empty (edge case), fall back to the rest
    if not nucleus:
        return onset, '', coda

    return onset, nucleus, coda


def _syllable_heavy(syl: str) -> bool:
    """True if syllable is heavy (long/diphthong nucleus OR has coda)."""
    _, nucleus, coda = _split_syllable(syl)
    if not nucleus:
        return False
    is_long = (nucleus in LONG_V_MAP.values() or
               len(nucleus) >= 2 and nucleus in DIPHTHONG)
    return is_long or bool(coda)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IPA SYLLABLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class IPASyllableGenerator:

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self._rng = random.Random(seed)

    def _rng(self):
        return self._rng

    # ── pick components ─────────────────────────

    def _pick_onset(self) -> str:
        r = self._rng.random()
        if r < 0.08:
            return ''                                     # empty onset
        if r < 0.45:
            return self._rng.choice(ALL_CONS)             # single C
        if r < 0.78:
            return self._rng.choice(ONSET_2)               # CC cluster
        return self._rng.choice(ONSET_3)                   # sCC cluster

    def _pick_nucleus(self) -> str:
        r = self._rng.random()
        if r < 0.52:
            return self._rng.choice(SHORT_V)               # short vowel
        if r < 0.78:
            return self._rng.choice(list(LONG_V_MAP.values()))  # long vowel
        return self._rng.choice(DIPHTHONG)                  # diphthong

    def _pick_coda(self) -> str:
        r = self._rng.random()
        if r < 0.22:
            return ''                                     # open syllable
        if r < 0.62:
            return self._rng.choice(ALL_CONS)             # single C coda
        return self._rng.choice(CODA_2)                    # CC cluster

    # ── diacritics (applied to components) ─────

    def _maybe_aspirate(self, onset: str) -> str:
        """Add ʰ to onset if first char is voiceless stop."""
        if not onset or len(onset) == 0:
            return onset
        first = onset[0]
        if first in STOPS_VL and self._rng.random() < 0.20:
            return onset[0] + 'ʰ' + onset[1:]
        return onset

    def _maybe_nasalize(self, nucleus: str, has_coda: bool) -> str:
        """Add nasalization ̃ only to vowels in open syllables."""
        if has_coda:
            return nucleus
        if nucleus in ALL_SHORT_V and self._rng.random() < 0.07:
            return nucleus + '̃'
        return nucleus

    # ── syllable ───────────────────────────────

    def make_syllable(self) -> str:
        """Build one phonologically valid IPA syllable."""
        onset = self._pick_onset()
        nucleus = self._pick_nucleus()
        coda = self._pick_coda()

        onset = self._maybe_aspirate(onset)
        nucleus = self._maybe_nasalize(nucleus, has_coda=bool(coda))

        return onset + nucleus + coda

    # ── stress placement ───────────────────────

    def _place_stress(self, syllables: List[str]) -> List[str]:
        """
        Attach stress markers (ˈ ˌ) to syllables following weight rules.
        Target: ~40% words have stress (aligns with TUM cross-domain distribution).
        - 1 syllable: bare (no marker)
        - 2 syllables: 40% stress on first; 60% bare
        - 3+ syllables: primary stress always; secondary stress 20% (was 50%)
        """
        n = len(syllables)
        if n == 1:
            return syllables

        weights = [_syllable_heavy(s) for s in syllables]

        if n == 2:
            # ~40% stress on first syllable — aligns with TUM ~38% stress rate
            if self._rng.random() < 0.40:
                if not weights[0] and weights[1] and self._rng.random() < 0.25:
                    return [syllables[0], 'ˈ' + syllables[1]]
                return ['ˈ' + syllables[0], syllables[1]]
            return syllables  # bare (no stress)

        # n >= 3
        # Primary: first heavy, else antepenult
        first_heavy = next((i for i, w in enumerate(weights) if w), 0)
        antepenult = max(0, n - 3)

        if weights[first_heavy] and first_heavy < antepenult:
            primary = antepenult
        else:
            primary = first_heavy

        result = list(syllables)
        result[primary] = 'ˈ' + result[primary]

        # Secondary: penult if heavy, 20% probability (was 50%)
        penult = n - 2
        if weights[penult] and self._rng.random() < 0.20:
            result[penult] = 'ˌ' + result[penult]

        return result

    # ── word ──────────────────────────────────

    def make_word(self,
                  minSyl: int = 1,
                  maxSyl: int = 4) -> str:
        """Build a multi-syllable IPA word with stress + IPA punctuation."""
        n = self._rng.randint(minSyl, maxSyl)
        syllables = [self.make_syllable() for _ in range(n)]
        if n >= 2:
            syllables = self._place_stress(syllables)
        word = ''.join(syllables)
        return self._apply_ipa_punctuation(word)

    # ── TUM IPA punctuation rules ─────────────────
    # Paired: [/] gloss, <> allomorph, () optional, // reconstruction
    # Single: . , : ; - ‿ (hyphen, glottal stop, etc.)
    PAIRED = {
        '[':  ('[', ']', 0.03),
        '<':  ('<', '>', 0.01),
        '(':  ('(', ')', 0.01),
        '/':  ('/', '/', 0.01),
    }
    SINGLE_PUNC = {
        '.':  ('after',  0.10),
        ',':  ('after',  0.10),
        ':':  ('after',  0.05),
        "'":  ('after',  0.05),
        '-':  ('middle', 0.50),
        'ʰ':  ('superscript', 0.10),  # aspiration as part of word
    }

    def _apply_ipa_punctuation(self, word: str) -> str:
        """Add IPA-typical punctuation: paired glosses, sentence-final, etc."""
        r = self._rng
        # Paired punctuation: wrap 1-3 words (only if multiple syllables)
        for opener, (op, cl, freq) in self.PAIRED.items():
            if r.random() < freq and len(word) > 3:
                # Wrap some middle portion
                span = r.randint(1, min(4, len(word) // 2 + 1))
                start = r.randint(0, max(0, len(word) - span))
                word = word[:start] + op + word[start:start+span] + cl + word[start+span:]
        # Sentence-final punctuation
        for mark, (placement, freq) in self.SINGLE_PUNC.items():
            if mark == '-':
                continue  # hyphen handled separately
            if r.random() < freq:
                word = word + mark
                break
        return word

    def make_word_set(self, count: int,
                      minSyl: int = 1,
                      maxSyl: int = 4) -> List[str]:
        """Generate `count` unique IPA words."""
        seen = set()
        while len(seen) < count:
            w = self.make_word(minSyl, maxSyl)
            if w not in seen:
                seen.add(w)
        return list(seen)


def make_technical_term(rng: random.Random) -> str:
    """Generate no-stress IPA technical/linguistic terms (mimics TUM cross-domain distribution)."""
    prefixes = ['nɒn-', 'supra-', 'alveo-', 'velo-', 'glot-', 'fric-',
                'næs-', 'læt-', 'voɪs-', 'vl̥', 'd̪', 'stʰ', 'krʰ',
                'kwɑ', 'pl̩', 'pr̩', 'n̩', 'm̩']
    roots    = ['ɾ', 'l', 'n', 'm', 'f', 'v', 's', 'z', 'ɣ', 'x', 'ʔ', 'ɬ']
    suffixes = ['əl', 'iə', 'aɪz', 'ʊs', 'ɛs', 'ʊm', 'oʊ', 'ɒn', 'ik', 'eɪt']
    parts = [rng.choice(prefixes), rng.choice(roots), rng.choice(suffixes)]
    if rng.random() < 0.10:
        diacritics = ['ː', 'ʰ', '̃', 'ʷ', 'ʲ']
        parts.append(rng.choice(diacritics))
    return ''.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RENDERING
# ─────────────────────────────────────────────────────────────────────────────

FONT_POOL = [
    # DejaVu family — 3 variants (sans, serif, mono)
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    # FreeFont family — 3 variants
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    # Ubuntu — sans + mono
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
    '/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf',
    # Liberation — metric-compatible with Arial/Times/Courier
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf',
    # Noto Mono — broad Unicode coverage
    '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
    '/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf',
    # Padauk — Burmese-derived, strong Unicode IPA coverage
    '/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf',
    # Project assets
    'assets/fonts/DejaVuSans.ttf',
]
FONT_POOL = [f for f in FONT_POOL if os.path.exists(f)]
if not FONT_POOL:
    FONT_POOL = [None]
print(f"[Font pool] {len(FONT_POOL)} fonts available")


def _bg_array(h: int, w: int, kind: str) -> np.ndarray:
    if kind == 'white':
        return np.full((h, w), 255, dtype=np.uint8)
    if kind == 'paper':
        base = np.full((h, w), 246, dtype=np.uint8)
        noise = np.random.randn(h, w) * 7
        return np.clip(base + noise, 0, 255).astype(np.uint8)
    if kind == 'gray':
        base = np.full((h, w), 198, dtype=np.uint8)
        noise = np.random.randn(h, w) * 14
        return np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.full((h, w), 255, dtype=np.uint8)  # fallback


# ── 强增强 ──────────────────────────────────────

def _elastic_deform(img: Image.Image, alpha: float = 30, sigma: float = 4) -> Image.Image:
    """Elastic deformation via Gaussian displacement fields."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha

    from scipy.ndimage import gaussian_filter, map_coordinates
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + dx, 0, w - 1)
    y_new = np.clip(y + dy, 0, h - 1)

    arr_def = map_coordinates(arr, [y_new.ravel(), x_new.ravel()], order=1).reshape(h, w)
    return Image.fromarray(np.clip(arr_def, 0, 255).astype(np.uint8))


def _random_erasing(img: Image.Image, p: float = 0.15) -> Image.Image:
    """Random rectangular occlusion."""
    if random.random() > p:
        return img
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    ew = random.randint(max(2, w // 12), max(4, w // 6))
    eh = random.randint(max(2, h // 12), max(4, h // 6))
    ex = random.randint(0, w - ew)
    ey = random.randint(0, h - eh)
    # Fill with background gray
    fill = random.randint(180, 255)
    arr[ey:ey+eh, ex:ex+ew] = fill
    return Image.fromarray(arr.astype(np.uint8))


def _perspective(img: Image.Image) -> Image.Image:
    """Small random perspective/shear distortion."""
    if random.random() > 0.35:
        return img
    w, h = img.size
    # 4 corners with small random offsets
    x0, y0 = 0, 0
    x1, y1 = w, 0
    x2, y2 = w, h
    x3, y3 = 0, h
    d = random.randint(0, max(2, w // 25))
    corners = [
        (x0 + random.randint(-d, d),   y0 + random.randint(-d, d)),
        (x1 + random.randint(-d, d),   y1 + random.randint(-d, d)),
        (x2 + random.randint(-d, d),   y2 + random.randint(-d, d)),
        (x3 + random.randint(-d, d),   y3 + random.randint(-d, d)),
    ]
    try:
        from PIL import ImageTransform
        return img.transform(
            (w, h), Image.Transform.PERSPECTIVE,
            ImageTransform.PERSPECTIVE_SPECIFICATION(
                list(sum(corners, ()))
            ),
            Image.Resampling.BICUBIC
        )
    except Exception:
        return img


def _albumentations_aug(img: Image.Image) -> Image.Image:
    """TUM-style augmentation via Albumentations: rotation + noise + brightness/contrast."""
    try:
        import albumentations as A
        import numpy as np
    except ImportError:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]

    transform = A.Compose([
        A.Rotate(limit=2, p=0.7, fill=255),
        A.GaussNoise(std_range=(0, 0.02), mean_range=(0, 0), p=0.6),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.08, 0.08),
            contrast_limit=(-0.08, 0.08),
            p=0.7,
        ),
    ])
    result = transform(image=arr)
    return Image.fromarray(result['image'])


def _stroke_variation(img: Image.Image) -> Image.Image:
    """Simulate stroke width variation via morphology."""
    if random.random() > 0.40:
        return img
    arr = np.array(img, dtype=np.float32)
    if random.random() < 0.5:
        from scipy.ndimage import grey_erosion
        arr = grey_erosion(arr, size=(1, 2))
    else:
        from scipy.ndimage import grey_dilation
        arr = grey_dilation(arr, size=(1, 2))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render(text: str, font_path, size: int, img_size: Tuple[int, int]) -> Image.Image:
    h, w = img_size
    bg = _bg_array(h, w, random.choice(['white', 'white', 'paper', 'gray']))
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = random.randint(4, 10)
    x = max(pad, (w - tw) // 2)
    y = max(pad, (h - th) // 2)
    draw.text((x, y), text, fill=0, font=font)

    # TUM-style: Albumentations first (rotation + noise + brightness/contrast)
    img = _albumentations_aug(img)
    # Then geometric
    img = _stroke_variation(img)
    img = _random_erasing(img)
    img = _perspective(img)

    return img


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.30:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr + np.random.randn(*arr.shape) * random.uniform(3, 9), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.1)))

    if random.random() < 0.12:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.82, 1.18))

    if random.random() < 0.08:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr * random.uniform(0.85, 1.15), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def _render_one(args):
    """Picklable worker: render one image."""
    (text, fname, font_path, fsize, do_aug, img_size) = args
    img = render(text, font_path, fsize, img_size)
    if do_aug:
        img = augment(img)
    return fname, img


def generate_dataset(
    output_dir: str,
    num_unique: int = 30000,
    samples_per_text: int = 5,
    img_size: Tuple[int, int] = (64, 256),
    min_syll: int = 1,
    max_syll: int = 4,
    val_split: float = 0.10,
    seed: int = 42,
    workers: int = None,
):
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    rng = random.Random(seed)
    print(f"Generating {num_unique} phonologically valid IPA strings…")
    gen = IPASyllableGenerator(seed=seed)
    all_words = gen.make_word_set(num_unique, minSyl=min_syll, maxSyl=max_syll)
    print(f"  {len(all_words)} unique strings generated")

    val_n = max(1, int(len(all_words) * val_split))
    rng.shuffle(all_words)
    val_words, train_words = all_words[:val_n], all_words[val_n:]
    print(f"  Train: {len(train_words)}, Val: {val_n}")

    for split, words in [('train', train_words), ('val', val_words)]:
        out_path = Path(output_dir) / split
        img_dir = out_path / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)

        # Build task list
        tasks = []
        for w_idx, text in enumerate(words):
            for s_idx in range(samples_per_text):
                tasks.append((
                    text,
                    f"ipa_{w_idx:05d}_{s_idx:02d}.png",
                    rng.choice(FONT_POOL),
                    rng.randint(10, 52),  # TUM-style 10pt + variety up to 52pt
                    rng.random() < 0.45,
                    img_size,
                ))

        # fname → text lookup for labels (built before submitting)
        fname_to_text = {t[1]: t[0] for t in tasks}
        labels = {}
        done = 0
        total = len(tasks)
        t0 = time.time()
        chunk_size = 5000  # submit in chunks to avoid memory blow-up

        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = {}
            for start in range(0, total, chunk_size):
                chunk = tasks[start:start + chunk_size]
                for t in chunk:
                    fut = ex.submit(_render_one, t)
                    pending[fut] = t[1]  # store fname

            for fut in as_completed(pending):
                fname, img = fut.result()
                img.save(img_dir / fname, optimize=True)
                labels[fname] = fname_to_text[fname]
                done += 1
                if done % 5000 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  {split}: {done}/{total}  {rate:.0f}/s  ETA {eta:.0f}s")

        with open(out_path / 'labels.json', 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {total} images → {img_dir}")

        # Inject ~15% no-stress technical IPA terms (aligns with TUM cross-domain)
        tech_n = max(100, int(len(words) * 0.15))
        tech_rng = random.Random(seed + (hash(split) % 1000))
        tech_start = done
        for i in range(tech_n):
            text = make_technical_term(tech_rng)
            fname = f"tech_{i:05d}_{split[0]}.png"
            tasks.append((
                text,
                fname,
                tech_rng.choice(FONT_POOL),
                tech_rng.randint(10, 52),
                tech_rng.random() < 0.45,
                img_size,
            ))
            fname_to_text[fname] = text
            img = render(text, tech_rng.choice(FONT_POOL),
                         tech_rng.randint(10, 52), img_size)
            if tech_rng.random() < 0.45:
                img = augment(img)
            img.save(img_dir / fname, optimize=True)
            labels[fname] = text
            done += 1

        with open(out_path / 'labels.json', 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"  {split}: +{tech_n} no-stress technical terms injected")

    print(f"\nDone. Data in: {output_dir}/{{train,val}}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Phonologically correct IPA data generator')
    ap.add_argument('--output-dir',   type=str,  default='data')
    ap.add_argument('--num-unique',  type=int,  default=30000)
    ap.add_argument('--samples-per-text', type=int, default=5)
    ap.add_argument('--height',       type=int,  default=64)
    ap.add_argument('--width',        type=int,  default=256)
    ap.add_argument('--min-syll',     type=int,  default=1)
    ap.add_argument('--max-syll',     type=int,  default=4)
    ap.add_argument('--val-split',    type=float, default=0.10)
    ap.add_argument('--seed',         type=int,  default=42)
    args = ap.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_unique=args.num_unique,
        samples_per_text=args.samples_per_text,
        img_size=(args.height, args.width),
        min_syll=args.min_syll,
        max_syll=args.max_syll,
        val_split=args.val_split,
        seed=args.seed,
    )
