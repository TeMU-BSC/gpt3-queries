from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import numpy as np
import json
from pprint import pprint

PATH_MLSUM_SAMPLE = 'with_ca_mlsum_sample.json'

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

with open(PATH_MLSUM_SAMPLE, 'r') as f:
    mlsum_langs = json.load(f)

print('MLSUM SAMPLE')
count = {}
for lang in mlsum_langs:
    dataset = mlsum_langs[lang]
    count[lang] = []
    for e in tqdm(dataset):
        count[lang].append(len(tokenizer(e['text'] + e['summary'])['input_ids']))

print('Total', sum(map(sum, count.values())))

for e in count:
    count[e] = {'mean': np.mean(count[e]), 'std': np.std(count[e]), 'min': min(count[e]), 'max': max(count[e]),
                'total': sum(count[e]),
                'len': len(count[e])}

pprint(count)


if __name__ == '__main__':
    pass
