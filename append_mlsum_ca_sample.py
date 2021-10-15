import os
import pandas as pd
import json
import numpy as np
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
MLSUM_CA_TEST_VALIDATED_PATH = os.path.join('assets', 'mlsum_ca_test_validated.tsv')
MLSUM_SAMPLE_PATH = 'mlsum_sample.json'
SAMPLE_SIZE = 500
df = pd.read_csv(MLSUM_CA_TEST_VALIDATED_PATH, sep='\t')
ca_array = df.to_numpy()
ca_list = []
MAX_TOKENS = 2000
for e in ca_array:
    if isinstance(e[0], str) and isinstance(e[1], str) and len(e[0].split()) + len(e[1].split()) > 0:
        if len(tokenizer(e[0] + e[1])['input_ids']) <= MAX_TOKENS:
            ca_list.append({'text': e[0], 'summary': e[1]})
with open(MLSUM_SAMPLE_PATH, 'r') as f:
    mlsum_sample = json.load(f)

if 'ca' not in mlsum_sample:
    sampled_array = list(np.random.choice(ca_list, size=SAMPLE_SIZE, replace=False))
    mlsum_sample['ca'] = sampled_array

    with open('with_ca_' + MLSUM_SAMPLE_PATH, 'w') as f:
        json.dump(mlsum_sample, f)
