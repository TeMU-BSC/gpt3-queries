import os
import json 
from sentence_splitter import split_text_into_sentences

def process(summary, lang):
    if lang == 'tu':
        lang = 'tr'
    if '\n' in summary:
        chopped_summary = summary.index('\n')
        summary = summary[:chopped_summary]
    split_summary = split_text_into_sentences(text=summary, language=lang)
    summary_top_3 = ' '.join(split_summary[0:3])
    return summary_top_3

SRC_PATH = '/home/ona/Documents/Papers/gpt3_paper/gpt3-queries/data/mlsum_outputs'
langs = ['ca','de','en','es','tu']

for filedir in os.listdir(SRC_PATH):
    
    model = filedir.replace('_gpt_summaries.json','')
    model_path = '/home/ona/Documents/Papers/gpt3_paper/gpt3-queries/bert_score/'+model
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filepath = os.path.join(SRC_PATH, filedir)
    results = {}
    
    for lang in langs:
        results[lang] = {'ref': [], 'hyp': []}

    with open(filepath) as dataset_file:
        dataset = dataset_file.readlines()

    for article in dataset:
        article_json = json.loads(article)
        results[article_json['lang']]['ref'].append(article_json['summary_gt'].replace(' .\n','. ').replace('\n',' '))
        results[article_json['lang']]['hyp'].append(process(article_json['summary_model'],article_json['lang']))

    for lang in langs:
        out_refs = os.path.join(model_path, lang+'_refs.txt')
        out_refs = open(out_refs,'w')
        for sent in results[lang]['ref']:
            out_refs.write(sent+'\n')
        out_hyps = os.path.join(model_path, lang+'_hyps.txt')
        out_hyps = open(out_hyps,'w')
        for sent in results[lang]['hyp']:
            out_hyps.write(sent+'\n')