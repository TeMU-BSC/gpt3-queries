# Usage: python3 stats_mlsum.py mlsum_dataset.json sentences/words/chars
import argparse
import json
import pprint
import numpy as np 
from scipy import stats
from sentence_splitter import split_text_into_sentences
from collections import Counter

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

def get_novelty(dataset):
    novelty_list = []
    for article in dataset:
        text_words = article['text'].split()
        summary_words = article['summary'].split()
        novelty_words = [word for word in summary_words if word not in text_words] # return only words not present in the text
        novelty = len(novelty_words)*100/len(summary_words)
        novelty_list.append(novelty)
    avg = np.mean(novelty_list)
    std = np.std(novelty_list)
    mode = stats.mode(novelty_list)[0][0]
    return avg, std, mode

def get_compressionratio(dataset):
    comp = [len(article['text'].split())/len(article['summary'].split()) for article in dataset]
    avg = np.mean(comp)
    std = np.std(comp)
    mode = stats.mode(comp)[0][0]
    return avg, std, mode

def vocab_stats(dataset):
    all_words = ' '.join([' '.join([article['text'].split(),article['summary'].split()] for article in dataset])
    print(len(all_words))
    vocab = set(all_words)
    print(len(vocab))
    word_occurrences = Counter(all_words)
    over_ten = [word[0] for word in word_occurrences if word[1] >= 10]
    print(over_ten)
    return vocab, over_ten

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistics script for Summarization. It expects a json file')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('tokens', help='Specify if char level, word level or sentence level: chars, words, sentences')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = json.load(dataset_file)

    results = {'ca': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':'', 'novelty_avg':'', 'novelty_std':'', 'novelty_mode':'', 'comp_avg':'', 'comp_std':'', 'comp_mode':'', 'vocab':'', 'over_ten':''},
                'en': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':'', 'novelty_avg':'', 'novelty_std':'', 'novelty_mode':'', 'comp_avg':'', 'comp_std':'', 'comp_mode':'', 'vocab':'', 'over_ten':''},
                'es': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':'', 'novelty_avg':'', 'novelty_std':'', 'novelty_mode':'', 'comp_avg':'', 'comp_std':'', 'comp_mode':'', 'vocab':'', 'over_ten':''},
                'de': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':'', 'novelty_avg':'', 'novelty_std':'', 'novelty_mode':'', 'comp_avg':'', 'comp_std':'', 'comp_mode':'', 'vocab':'', 'over_ten':''},
                'tu': {'text_avg_len':'','text_std_len':'','text_mode_len':'','sum_avg_len':'','sum_std_len':'','sum_mode_len':'', 'novelty_avg':'', 'novelty_std':'', 'novelty_mode':'', 'comp_avg':'', 'comp_std':'', 'comp_mode':'', 'vocab':'', 'over_ten':''},
    }
    for lang in dataset:
        print(lang)
        #results[lang]['text_avg_len'], results[lang]['text_std_len'], results[lang]['text_mode_len'] = get_stats(dataset[lang], 'text', args.tokens, lang)
        #results[lang]['sum_avg_len'], results[lang]['sum_std_len'], results[lang]['sum_mode_len'] = get_stats(dataset[lang], 'summary', args.tokens, lang)
        #results[lang]['novelty_avg'], results[lang]['novelty_std'], results[lang]['novelty_mode'] = get_novelty(dataset[lang])
        #results[lang]['comp_avg'], results[lang]['comp_std'], results[lang]['comp_mode'] = vocab_stats(dataset[lang])
        results[lang]['vocab'], results[lang]['over_ten'] = get_compressionratio(dataset[lang])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    for lang in results:
        print(";;;".join([str(results[lang][score]).replace('.',',') for score in results[lang]]))
        