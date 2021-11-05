import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
# Don't know why turksih is underneath
palette = 'rocket'
labels=['tu', 'ru','ca','de','es','en','EM']

plt.rc('font', **font)
#PLOT QA F1SCORE
f1score = pd.read_csv('data/qa_F1.tsv', sep='\t')
sns.barplot(data=f1score, x='Model', y='Score', hue='Language', palette=palette)

plt.savefig('figures/qa_scores.png')
plt.close()
