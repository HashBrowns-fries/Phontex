"""TrOCR quick sanity test"""
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Use generation_config for generation settings
model.generation_config.max_length = 64
model.generation_config.early_stopping = True
model.generation_config.num_beams = 4
model.generation_config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

device = torch.device('cuda')
model.to(device)
model.eval()

with open('data/val/labels.json') as f:
    labels_dict = json.load(f)

samples = [
    ('data/val/images/ipa_0000_00.png', labels_dict.get('ipa_0000_00.png', '')),
    ('data/val/images/ipa_0000_01.png', labels_dict.get('ipa_0000_01.png', '')),
    ('data/val/images/ipa_0000_02.png', labels_dict.get('ipa_0000_02.png', '')),
]

for img_path, gt in samples:
    pixel_values = processor(Image.open(img_path).convert('RGB'), return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        gen_ids = model.generate(pixel_values)
    text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(f'GT: {gt!r}')
    print(f'PD: {text!r}')
    print()
