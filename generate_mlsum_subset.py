from datasets import load_dataset
import datasets
from transformers import GPT2Tokenizer, RobertaTokenizer, AutoTokenizer, BertTokenizerFast, EncoderDecoderModel, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
import numpy as np
import torch
mlsum_langs = ["de", "es", "tu"]  # "ru", "tu"]

SAMPLE_SIZE = 500
SEED = 42
np.random.seed(SEED)

MAX_TOKENS = 2000

#summarization models per language
MODELS_DICT = {
    'de': 'T-Systems-onsite/mt5-small-sum-de-en-v2',
    'en': 'T-Systems-onsite/mt5-small-sum-de-en-v2',
    'es': 'Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization',
    'tu': 'mrm8488/bert2bert_shared-turkish-summarization',
}

rouge = datasets.load_metric('rouge')

def get_prediction(text,ckpt,lang):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if lang in ['en','de']:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
        inputs = tokenizer([text], return_tensors="pt")
    else:
        if lang == 'tu':
            tokenizer = BertTokenizerFast.from_pretrained(ckpt)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(ckpt)
        model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
        inputs = tokenizer([text],  padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    prediction =  tokenizer.decode(output[0], skip_special_tokens=True)
    return(prediction)

def get_rouge(gt,prediction):
    results = rouge.compute(predictions=[prediction], references=[gt])
    return(results["rouge1"].mid.fmeasure)

sampled = {}
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
dataset.shuffle(seed=SEED)
en_dataset = []
for e in tqdm(dataset):
    if len(tokenizer(e['article'] + e['highlights'])['input_ids']) <= MAX_TOKENS:
        prediction = get_prediction(e['article'], MODELS_DICT['en'],'en')
        rouge_score = get_rouge(e['highlights'],prediction)
        if rouge_score >= 0.1:
            en_dataset.append({'text': e['article'], 'summary': e['highlights']})
        if len(en_dataset) == 500:
            break
#sampled['en'] = list(np.random.choice(en_dataset, size=SAMPLE_SIZE, replace=False))
sampled['en'] = en_dataset
for lang in mlsum_langs:
    dataset = load_dataset("mlsum", lang, split='test')
    dataset.shuffle(seed=SEED)
    lang_dataset = []
    for e in tqdm(dataset):
        if len(tokenizer(e['text'] + e['summary'])['input_ids']) <= MAX_TOKENS:
            prediction = get_prediction(e['text'], MODELS_DICT[lang],lang)
            rouge_score = get_rouge(e['summary'],prediction)
            if rouge_score >= 0.1:
                lang_dataset.append({'text': e['text'], 'summary': e['summary']})
            if len(lang_dataset) == 500:
                break
    sampled[lang] = lang_dataset#list(np.random.choice(lang_dataset, size=SAMPLE_SIZE, replace=False))

with open('mlsum_sample.json', 'w') as f:
    json.dump(sampled, f)

