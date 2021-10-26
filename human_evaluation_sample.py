from datasets import load_dataset
import random
import json
import os
#Select a sample of 20 articles for human evaluation

SEED = 42
random.seed(SEED)

def process_sample(sample,lang):
    for line in sample:
        gpt_keys = {'ada': '', 'babbage':'', 'curie':'', 'davinci':''}
        line.update(gpt_keys)

    PATH = os.path.join('data','human_eval',lang+'.json')
    with open(PATH, 'w') as f:
        for element in sample:
            json.dump(element,f)
            f.write('\n')
    
if __name__ == '__main__':
    summaries = {}
    gt = {}
    # load english dataset
    en_dataset = load_dataset("cnn_dailymail", '3.0.0', split='test')
    en_sample = random.choices(en_dataset,k=20)
    # rename dict keys
    for entry in en_sample:
        entry["text"] = entry.pop("article")
        entry["summary"] = entry.pop("highlights")
        # replace new lines by whitespace
        entry["summary"] = entry["summary"].replace(' .\n','. ')
    # load catalan dataset
    ca_dataset = [json.loads(line) for line in open('data/mlsum_ca.json','r').readlines()]
    ca_sample = random.choices(ca_dataset,k=20)
    # save data to file
    process_sample(en_sample, 'en')
    process_sample(ca_sample, 'ca')