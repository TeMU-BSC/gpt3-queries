# Usage: python3 stats_mlsum.py mlsum_dataset.json sentences/words/chars
import argparse
import json
import pprint
import numpy as np 
from scipy import stats
from sentence_splitter import split_text_into_sentences

def get_stats(dataset, type, tokens, lang):
    if tokens == 'words':
        len_list = [len(article[type].split()) for article in dataset]
    elif tokens == 'chars':
        len_list = [len(article[type]) for article in dataset]
    else:
        if lang == 'tu':
            lang = 'tr'
        len_list = [len(split_text_into_sentences(text=article[type], language=lang)) for article in dataset]
    avg = np.mean(len_list)
    std = np.std(len_list)
    mode = stats.mode(len_list)[0][0]
    return avg, std, mode
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistics script for Summarization. It expects a json file')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('tokens', help='Specify if char level, word level or sentence level: chars, words, sentences')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = json.load(dataset_file)

    results = {'ca': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':''},
                'en': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':''},
                'es': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':''},
                'de': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':''},
                'tu': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':''}
    }
    for lang in dataset:
        results[lang]['text_avg_len'], results[lang]['text_std_len'], results[lang]['text_mode_len'] = get_stats(dataset[lang], 'text', args.tokens, lang)
        results[lang]['sum_avg_len'], results[lang]['sum_std_len'], results[lang]['sum_mode_len'] = get_stats(dataset[lang], 'summary', args.tokens, lang)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    for lang in results:
        print(";;;".join([str(results[lang][score]).replace('.',',') for score in results[lang]]))
        