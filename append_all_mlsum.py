import os
import pandas as pd
import json
import numpy as np
from transformers import GPT2TokenizerFast
MLSUM_SAMPLES_FOLDER = 'data/mlsum_samples'
MLSUM_SAMPLE_PATH = 'mlsum_sample.json'

mlsum_sample={}
for filename in os.listdir(MLSUM_SAMPLES_FOLDER):
   with open(os.path.join(MLSUM_SAMPLES_FOLDER, filename), 'r') as f: 
       lang_sample = json.load(f)
       lang = list(lang_sample.keys())[0]
       mlsum_sample[lang] = lang_sample[lang]
    
with open(MLSUM_SAMPLE_PATH, 'w') as f:
    json.dump(mlsum_sample,f)