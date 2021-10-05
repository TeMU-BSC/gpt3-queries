from datasets import load_dataset
from transformers import GPT2TokenizerFast
import json
from tqdm import tqdm
import numpy as np
mlsum_langs = ["de", "es", "ru", "tu"]

SAMPLE_SIZE = 500
SEED = 42
np.random.seed(SEED)

sampled = {}

for lang in mlsum_langs:
    dataset = load_dataset("mlsum", lang, split='test')
    lang_dataset = []
    for e in tqdm(dataset):
        lang_dataset.append({'text': e['text'], 'summary': e['summary']})
    sampled[lang] = list(np.random.choice(lang_dataset, size=SAMPLE_SIZE, replace=False))

with open('mlsum_sample.json', 'w') as f:
    json.dump(sampled, f)

