import sys
import json
import pandas as pd
import os
import csv 
from sentence_splitter import split_text_into_sentences
import random
import numpy as np

def process(summary, lang):
    if lang == 'tu':
        lang = 'tr'
    if '\n' in summary:
        chopped_summary = summary.index('\n')
        summary = summary[:chopped_summary]
    split_summary = split_text_into_sentences(text=summary, language=lang)
    summary_top_3 = ' '.join(split_summary[0:3])
    return summary_top_3

if __name__ == '__main__':
    lang = sys.argv[1]
    models = ['ada', 'babbage', 'curie', 'davinci']
    model_sums = {}
    for model in models:
        discarded_texts = []
        PATH = os.path.join('data','mlsum_outputs',model+'_gpt_summaries.json')
        #models = {'human': [], 'ada': [], 'babbage': [], 'curie': [], 'davinci': []}
        with open(PATH, 'r') as f:
            for article in f:
                article = json.loads(article)
                if article['lang'] == lang:
                    if article['text'] in model_sums.keys():
                        if article['summary_model'] not in list(model_sums[article['text']].values()): #force different summaries per model
                            model_sums[article['text']][model] = process(article['summary_model'], lang)
                        else: 
                            del model_sums[article['text']]
                            discarded_texts.append(article['text'])
                    else:
                        if article['text'] not in discarded_texts:
                            model_sums[article['text']] = {'human':'', 'ada':'', 'babbage':'', 'curie':'', 'davinci':''}
                            model_sums[article['text']]['human'] = article['summary_gt'].replace(' .\n','. ').replace('\n',' ')
                            model_sums[article['text']][model] = process(article['summary_model'], lang)
            
        final_list = []
        text_count = 0
        for entry in model_sums:
            if text_count < 76:
                text_ID = text_count
                text = entry
                final_list.append([text_ID,text,'text'])
                sum_count = 1
                random_order = random.sample(list(model_sums[entry].keys()), 5) #randomly sample keys
                for model in random_order:
                    sum_ID = str(text_count) + '.' + str(sum_count)
                    summary = model_sums[entry][model]
                    final_list.append([sum_ID,summary,model])
                    sum_count += 1
                text_count += 1

    OUT_PATH = '/home/ona/Documents/Papers/gpt3_paper/gpt3-queries/data/human_eval/summary_eval/'
    all_articles = pd.DataFrame(final_list, columns=['ID','Text/Summary','model'])
    all_articles.to_csv(os.path.join(OUT_PATH,lang+'_all_articles_dedup.tsv'), sep='\t')