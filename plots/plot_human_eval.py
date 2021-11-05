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
palette = ['#0082a5','#fb9800']
labels=['tu', 'ru','ca','de','es','en','EM']

plt.rc('font', **font)
sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
#PLOT CATALAN
ca_eval = pd.read_csv('data/human_ca.tsv', sep='\t')
#https://www.pythoncharts.com/python/stacked-bar-charts/
sns.barplot(data=ca_eval,x='Model',y='AI', color=palette[0])
sns.barplot(data=ca_eval, x='Model', y='Human', color=palette[1])
#Create legend with corresponding colors
plt.legend(loc='upper left', labels=['Human','AI'])
ax = plt.gca()
ax.set_ylim([0, 60])
ax.set(xlabel='Model', ylabel='Sentences')
leg = ax.get_legend()
for handle in enumerate(leg.legendHandles):
        index = handle[0]
        leg.legendHandles[index].set_color(sns.color_palette(palette).as_hex()[index])
        leg.legendHandles[index].set_marker("o")

plt.savefig('figures/human_ca.png')
plt.close()
