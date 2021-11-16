import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mtick

matplotlib.rcParams['legend.handlelength'] = 0
matplotlib.rcParams['legend.numpoints'] = 1
sns.set_style("white")

sns.set(rc={'figure.figsize':(8,4.5)})
font = {'family' : 'Sans Serif',
        'size'   : 12}
plt.rc('font', **font)

palette = ['#0082a5','#fb9800']

#df = pd.read_csv('data/human_eval.tsv', sep='\t')
df = pd.read_csv('data/human_eval_percentage.tsv', sep='\t')
#READ DFs
ca_eval = df[df['Language'] == 'ca']
en_eval = df[df['Language'] == 'en']

fig, axs = plt.subplots(1,2)

sns.barplot(data=en_eval,x='Model',y='AI', color=palette[0], ax=axs[0])
sns.barplot(data=en_eval, x='Model', y='Human', color=palette[1],ax=axs[0])
axs[0].set_ylim([0, 100])
axs[0].set_ylabel('Sentences', weight='bold')
axs[0].set_xlabel(xlabel='English',weight='bold')
axs[0].yaxis.set_major_formatter(mtick.PercentFormatter())
sns.barplot(data=ca_eval,x='Model',y='AI', color=palette[0], ax=axs[1])
sns.barplot(data=ca_eval, x='Model', y='Human', color=palette[1], ax=axs[1])
#Create legend with corresponding colors
axs[1].set_ylim([0, 100])
axs[1].set(ylabel='')
axs[1].set_xlabel(xlabel='Catalan',weight='bold')
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())

#LEGEND

human = mlines.Line2D([], [], color='#fb9800', marker='o', linestyle='None',
                          markersize=7, label='Human')

ai = mlines.Line2D([], [], color='#0082a5', marker='o', linestyle='None',
                          markersize=7, label='AI')
fig.legend(handles=[human,ai], loc='upper center', frameon=False)
#sns.color_palette(palette).as_hex()0
plt.savefig('figures/human_eval.pdf')
plt.close()