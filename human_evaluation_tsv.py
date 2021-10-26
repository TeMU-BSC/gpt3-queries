import random
import json
import os
from sentence_splitter import split_text_into_sentences
import sys
import pandas as pd

# Select a sample of 60 sentences (deduped)
# With that same headline, generate sentences with GPT-3 and select a sample of 60 sentences
# Randomize sentences, all annotate 120 sentence
#['de','es','ru','tu']
SEED = 42
random.seed(SEED)

def get_random_sample(text, lang):
    sentences = split_text_into_sentences(text=text, language=lang)
    # Deduplicate sentences
    sentences_dedup = list(set(sentences))
    sentences_sample = random.choices(sentences_dedup, k=60)
    return sentences_sample

if __name__ == '__main__':
    lang = sys.argv[1]
    PATH = os.path.join('data','human_eval',lang+'.json')
    models = {'human': '', 'ada':'', 'babbage':'', 'curie':'', 'davinci':''}
    with open(PATH, 'r') as f:
        for element in f:
            element = json.loads(element)
            element["human"] = element.pop("text")
            for model in models:
                models[model] = '. '.join([models[model],element[model]])
    
    human_eval = pd.DataFrame()
    for model in models:
        sample = get_random_sample(models[model],lang)
        sample_model = list(zip(sample, [model]*60))
        human_eval = human_eval.append(sample_model, ignore_index=True)
    human_eval = human_eval.rename(columns={0:'sentence',1:'model'})
    # random
    human_eval_random = human_eval.sample(frac=1)

    human_eval_tsv = os.path.join('data','human_eval',lang+'.tsv')
    human_eval_random.to_csv(human_eval_tsv)

