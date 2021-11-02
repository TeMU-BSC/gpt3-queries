import random
import json
import os
import string

from sentence_splitter import split_text_into_sentences
import sys
import pandas as pd
import re
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
# Select a sample of 60 sentences (deduped)
# With that same headline, generate sentences with GPT-3 and select a sample of 60 sentences
# Randomize sentences, all annotate 120 sentence
#['de','es','ru','tu']
SEED = 42
random.seed(SEED)


def normalize(sentence, lang):
    sentence = ' ' + ' '.join(sentence.split()) + ' '
    if lang in ['ca']:
        consonants_apost = '|'.join(list("dlmnst") + list('DLMNST'))
        vocals = '|'.join(list('aeiou' + 'AEIOU' + 'àèéíòóúïü' + 'ÀÈÉÍÒÓÚÏÜ'))
        sentence = re.sub(f"(-| )({consonants_apost})( )*(')( )*(h|H)?( )*({vocals})", r'\1\2\3__APOST__\5\6\7\8', sentence)
        sentence = sentence.replace("'", '"')
        sentence = sentence.replace('__APOST__', "'")
    sentence = re.sub(r'(\w| )(\.\.\. )', r'\1… ', sentence)
    for quote in ['"', "“", "”", "‘", "’", "«", "»", "‹", "›", "„", "“"]:
        sentence.replace(quote, '"')
    while True:
        new_sentence = sentence
        for c in string.punctuation:
            new_sentence = new_sentence.replace(c + c, c)
        if new_sentence == sentence:
            break
        sentence = new_sentence
    sentence = sentence.replace('…', '...')
    sentence = sentence.strip()
    mt = MosesTokenizer(lang)
    mdt = MosesDetokenizer(lang)
    mn = MosesPunctNormalizer(lang)
    sentence = mn.normalize(sentence)
    sentence = mdt.detokenize(mt.tokenize(sentence, aggressive_dash_splits=False, escape=False), unescape=False)
    if lang in ['ca']:
        consonants_apost = '|'.join(list("dlmnst") + list('DLMNST'))
        vocals = '|'.join(list('aeiou' + 'AEIOU' + 'àèéíòóúïü' + 'ÀÈÉÍÒÓÚÏÜ'))
        sentence = re.sub(f"({consonants_apost})( )(')(h|H)?({vocals})", r'\1\3\4\5', sentence)
    return sentence


def get_random_sample(text, lang, samples_per_article):
    sentences = split_text_into_sentences(text=text, language=lang)
    # Remove sentences that don't contain letters
    sentences_clean = [sentence for sentence in sentences if re.match(r'[a-zA-Z]{2,}',sentence)]
    # Filter sentences that are smaller than ten words and smaller than 40
    sentences_filtered = [sentence for sentence in sentences_clean if len(sentence.split())>=10 and len(sentence.split())<=40]
    # Normalize punctuation
    sentences_norm = [normalize(sentence, lang) for sentence in sentences_filtered]
    # Random sample 3 sentences
    try:
        sentence_sample = random.choices(sentences_norm, k=3)
    except:
        sentence_sample = ['']
    return sentence_sample

if __name__ == '__main__':
    lang = sys.argv[1]
    PATH = os.path.join('data','human_eval',lang+'.json')
    models = {'human': [], 'ada': [], 'babbage': [], 'curie': [], 'davinci': []}
    with open(PATH, 'r') as f:
        for article in f:
            article = json.loads(article)
            article["human"] = article.pop("text")
            for model in models:
                models[model].append(article[model])

    human_eval = pd.DataFrame()
    for model in models:
        samples_per_article = []
        for article in models[model]:
            sentence_sample = get_random_sample(article, lang, samples_per_article)
            samples_per_article.extend(sentence_sample)
        sample_model = list(zip(samples_per_article, [model]*60))
        human_eval = human_eval.append(sample_model, ignore_index=True)
    human_eval = human_eval.rename(columns={0:'sentence',1:'model'})

    # random
    #human_eval_random = human_eval.sample(frac=1)
    #human_eval_random.to_csv(human_eval_tsv,sep='\t')

    human_eval_tsv = os.path.join('data','human_eval',lang+'_test.tsv')
    human_eval.to_csv(human_eval_tsv,sep='\t')
