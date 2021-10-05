from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import numpy as np

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

xquad_langs = ['xquad.de', 'xquad.es', 'xquad.ru', 'xquad.tr']#['xquad.ar', 'xquad.de', 'xquad.zh', 'xquad.vi', 'xquad.en', 'xquad.es', 'xquad.hi', 'xquad.el',
              # 'xquad.th', 'xquad.tr', 'xquad.ru', 'xquad.ro']

mlsum_langs = ["de", "es", "ru", "tu"] #["de", "es", "fr", "ru", "tu"]
ESTIMATE_XQUAD = True
ESTIMATE_MLSUM = True

if ESTIMATE_XQUAD:
    print('XQUAD')
    count = {}
    for xquad_lang in tqdm(xquad_langs):
        xquad = load_dataset('xquad', xquad_lang, split='validation')
        count[xquad_lang] = 0
        for e in tqdm(xquad):
            count[xquad_lang] += len(tokenizer(e['context'] + e['question'])['input_ids'])
    count['xquad.ca'] = 0
    dataset = load_dataset("BSC-TeMU/xquad-ca", split='test')
    for e in tqdm(dataset):
        count['xquad.ca'] += len(tokenizer(e['context'] + e['question'])['input_ids'])
    from pprint import pprint
    pprint(count)
    total = 0
    for e in count:
        total += count[e]
    print('Total:', total)

if ESTIMATE_MLSUM:
    print('MLSUM')
    count = {}
    for lang in mlsum_langs:
        dataset = load_dataset("mlsum", lang, split='test')
        count[lang] = []
        for e in tqdm(dataset):
            count[lang].append(len(tokenizer(e['text'] + e['summary'])['input_ids']))
    for e in count:
        count[e] = {'mean': np.mean(count[e]), 'std': np.std(count[e]), 'min': min(count[e]), 'max': max(count[e]),
                    'len': len(count[e])}
    pprint(count)
    total = 0
    #for e in count:
    #    total += count[e]
    #print('Total:', total)

if __name__ == '__main__':
    pass
