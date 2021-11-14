import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
# Don't know why turksih is underneath
palette = 'muted'
labels=['ca', 'de', 'es', 'en', 'tu', 'ru']

plt.rc('font', **font)
# x axis
x_axis_dict = {'ada': math.log(350), 'babbage' : math.log(1300), 'curie' : math.log(6700), 'davinci' : math.log(175000)}

#PLOT QA F1SCORE
df = pd.read_csv('data/qa_EM.tsv', sep='\t')
df['x_axis'] = df['Model'].map(x_axis_dict) 
df['x_axis'] = df['x_axis'].apply(lambda x: round(x,0))
sns.lineplot(data=df, x='x_axis', y='Score', hue='Language', style='Language',markers=True, palette=palette, hue_order=labels)

#PLOT QA EM
#exact_match = pd.read_csv('data/qa_EM.tsv', sep='\t')
#sns.pointplot(data=exact_match, x='Model', y='Score', hue='Language', markers='^', dodge=0.7, color='black', linestyles='',  facet_kws={'legend_out': True})#, axx2)
ax = plt.gca()
ax.set_ylim([-0.01, 0.4])
ax.set(xlabel = 'Model', ylabel='EM', xticklabels=['', 'ada','babbage','','curie','','','davinci'])
plt.legend(loc='best')
plt.savefig('figures/qa_EM.pdf')
plt.close()
