# From: https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np

ARTICLES_DICT = {'ca': ['un','una','uns','unes','el','la','els','les'],
     'es': ['uno','una','unos','unas','el','la','los','las'],
     'ru': [],
     'de': ['ein','eine','einen','einem','einer','eines','der','das','die','den','dem','der','den','des'],
     'en': ['a','an','the']
    }

def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text, lang):
        if lang in ARTICLES_DICT:
            regex = r'\b('+"|".join(ARTICLES_DICT[lang])+r')\b'
            text = re.sub(regex, ' ', text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)),lang))


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return (normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang))


def evaluate(dataset):
    langs = ['de','en','es','global'] #'ru', 'tu', 'ca'
    results = {}
    for lang in langs:
        results[lang] = {'f1': [], 'exact_match': []}
    for question in dataset:
        lang = question['lang'][-2:]
        prediction = list(question.values())[0]
        gt = question['question']['true_answer']
        results[lang]['f1'].append(f1_score(prediction, gt, lang))
        results[lang]['exact_match'].append(exact_match_score(prediction, gt, lang))
        results['global']['f1'].append(f1_score(prediction, gt, lang))
        results['global']['exact_match'].append(exact_match_score(prediction, gt, lang))
    for lang in langs:
        results[lang]['f1'] = np.mean(results[lang]['f1'])
        results[lang]['exact_match'] =  np.mean(results[lang]['exact_match'])
    return results


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = [json.loads(line) for line in dataset_file.readlines()]
    print(json.dumps(evaluate(dataset)))
