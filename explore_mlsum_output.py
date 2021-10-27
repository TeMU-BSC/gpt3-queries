import sys
import json
import pandas as pd
from evaluate_rouge import process
lang = sys.argv[1]
model = sys.argv[2]
f = open('/home/ona/Documents/Papers/gpt3_paper/gpt3-queries/data/mlsum_outputs/'+model+'_gpt_summaries.json','r')

summaries = []

for entry in f.readlines():
    article = json.loads(entry)
    if article['lang'] == lang:
        summaries.append(article)
gt_summary = [article['summary_gt'] for article in summaries]
model_summary = [process(article['summary_model'], lang) for article in summaries]
text = [article['text'] for article in summaries]
df = pd.DataFrame(zip(gt_summary,model_summary,text))
out = open(model+'_'+lang+'.tsv','w')
df.to_csv(out, sep='\t')