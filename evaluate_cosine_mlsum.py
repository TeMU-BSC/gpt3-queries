# Usage example: python3 evaluate_rouge.py data/gpt_summaries.json 
import argparse
import json
import datasets
from itertools import chain 
import pprint
import numpy as np 
from sentence_splitter import split_text_into_sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def process(summary, lang):
    if lang == 'tu':
        lang = 'tr'
    if '\n' in summary:
        chopped_summary = summary.index('\n')
        summary = summary[:chopped_summary]
    split_summary = split_text_into_sentences(text=summary, language=lang)
    summary_top_3 = ' '.join(split_summary[0:3])
    return summary_top_3
    

def cosine(ground_truth,prediction):
    gt_vect = model.encode(ground_truth)
    pred_vect = model.encode(prediction)
    cos = cosine_similarity(gt_vect.reshape(1, -1),pred_vect.reshape(1, -1))
    return cos

def evaluate(dataset):
    langs = ["ca","de",'en','es','tu','global']
    processed_data = {}
    results = {}
    for lang in langs:
        processed_data[lang] = {'cos': []}
        results[lang] = {'cos_avg':'', 'cos_std':''}
    for article in dataset:
        article_json = json.loads(article)
        article_json['summary_gt'] = article_json['summary_gt'].replace(' .\n','. ')
        article_json['summary_gt'] = article_json['summary_gt'].replace('\n',' ')
        article_json['summary_model'] = process(article_json['summary_model'], article_json['lang'])
        processed_data[article_json['lang']]['cos'].append(cosine(article_json['summary_gt'],article_json['summary_model']))

    for lang in langs:
        results[lang]['cos_avg'] = np.mean(processed_data[lang]['cos'])
        results[lang]['cos_std'] = np.std(processed_data[lang]['cos'])
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script for Summarization. It expects a file where each line is a dictionary with the following entries: text, summary_gt, summary_model, lang')
    parser.add_argument('dataset_file', help='Dataset file')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = dataset_file.readlines()
    pp = pprint.PrettyPrinter(indent=4)
    results = evaluate(dataset)
    pp.pprint(results)
    for score in ['cos_avg','cos_std']:
        scores = []
        print(score)
        for lang in ["ca","de",'en','es','tu','global']:
                scores.append(str(results[lang][score]).replace('.',','))
        print(';'.join(scores))
