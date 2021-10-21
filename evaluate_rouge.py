# Usage example: python3 evaluate_rouge.py data/gpt_summaries.json 
import argparse
import json
import datasets
from itertools import chain 
import pprint
import numpy as np 

rouge = datasets.load_metric('rouge')

def get_rouge(ground_truths,predictions):
    rouge_score = rouge.compute(predictions=predictions, references=ground_truths)
    rouge_score = rouge_score['rouge1'].mid
    rouge_score = [round(score,3) for score in rouge_score]
    return rouge_score

def evaluate(dataset):
    langs = ["ca","de",'en','es','tu','global']
    processed_data = {}
    results = {}
    for lang in langs:
        processed_data[lang] = {"ground_truths":[],"predictions": [], "lengths_gt": [], "lengths_model":[]}
        results[lang] = {'rouge':{}, 'summary_lengths': {}}
    for article in dataset:
        article_json = json.loads(article)
        processed_data[article_json['lang']]['ground_truths'].append(article_json['summary_gt'])
        processed_data[article_json['lang']]['predictions'].append(article_json['summary_model'])
        processed_data[article_json['lang']]['lengths_gt'].append(len(article_json['summary_gt'].split()))
        processed_data[article_json['lang']]['lengths_model'].append(len(article_json['summary_model'].split()))
        processed_data['global']['ground_truths'].append(article_json['summary_gt'])
        processed_data['global']['predictions'].append(article_json['summary_model'])
        processed_data['global']['lengths_gt'].append(len(article_json['summary_gt'].split()))
        processed_data['global']['lengths_model'].append(len(article_json['summary_model'].split()))

    for lang in langs:
        lengths_gt = processed_data[lang]['lengths_gt']
        lengths_model = processed_data[lang]['lengths_model']
        precision, recall, fmeasure = get_rouge(processed_data[lang]['ground_truths'],processed_data[lang]['predictions'])
        results[lang]['summary_lengths'] = {'avg_gt': np.mean(lengths_gt), 'std_gt': np.std(lengths_gt),  'avg_model': np.mean(lengths_model), 'std_model': np.std(lengths_model)}
        results[lang]['rouge'] = {'precision':precision,'recall':recall,'fmeasure':fmeasure}
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script for Summarization. It expects a file where each line is a dictionary with the following entries: text, summary_gt, summary_model, lang')
    parser.add_argument('dataset_file', help='Dataset file')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = dataset_file.readlines()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(evaluate(dataset))