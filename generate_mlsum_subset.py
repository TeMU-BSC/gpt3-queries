from datasets import load_dataset
from transformers import GPT2TokenizerFast
import json
from tqdm import tqdm
import numpy as np
mlsum_langs = ["de", "es", "tu"]  # "ru", "tu"]

SAMPLE_SIZE = 500
SEED = 42
np.random.seed(SEED)

MAX_TOKENS = 2000

sampled = {}
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
en_dataset = []
for e in dataset:
    if len(tokenizer(e['article'] + e['highlights'])['input_ids']) <= MAX_TOKENS:
        en_dataset.append({'text': e['article'], 'summary': e['highlights']})
sampled['en'] = list(np.random.choice(en_dataset, size=SAMPLE_SIZE, replace=False))

for lang in mlsum_langs:
    dataset = load_dataset("mlsum", lang, split='test')
    lang_dataset = []
    for e in tqdm(dataset):
        if len(tokenizer(e['text'] + e['summary'])['input_ids']) <= MAX_TOKENS:
            lang_dataset.append({'text': e['text'], 'summary': e['summary']})
    sampled[lang] = list(np.random.choice(lang_dataset, size=SAMPLE_SIZE, replace=False))

with open('mlsum_sample.json', 'w') as f:
    json.dump(sampled, f)

