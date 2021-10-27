import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['legend.handlelength'] = 0
matplotlib.rcParams['legend.numpoints'] = 1
sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
# Don't know why turksih is underneath
palette = 'rocket'
labels=['tu', 'ru','ca','de','es','en','EM']

plt.rc('font', **font)
#PLOT QA F1SCORE
f1score = pd.read_csv('qa_F1.tsv', sep='\t')
sns.barplot(data=f1score, x='Model', y='Score', hue='Language', palette=palette)

#PLOT QA EM
exact_match = pd.read_csv('qa_EM.tsv', sep='\t')
sns.pointplot(data=exact_match, x='Model', y='Score', hue='Language', markers='^', dodge=0.7, color='black', linestyles='',  facet_kws={'legend_out': True})#, axx2)

#Create legend with corresponding colors
plt.legend(loc='upper left', labels=labels)
ax = plt.gca()
ax.set_ylim([-0.01, 0.5])
leg = ax.get_legend()
for handle in enumerate(leg.legendHandles):
        index = handle[0]
        leg.legendHandles[index].set_color(sns.color_palette(palette).as_hex()[index])
        leg.legendHandles[index].set_marker("o")

leg.legendHandles[-1].set_marker('^')
leg.legendHandles[-1].set_color('black')
plt.savefig('plot_qa_scores.png')
plt.close()
